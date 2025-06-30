#!/usr/bin/env python3
"""
Multi-architecture container builder with retry logic and cache management.
This script orchestrates buildah operations for building multi-arch images.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)


class IIBError(Exception):
    """Custom exception for IIB operations."""
    pass


class ExternalServiceError(IIBError):
    """Exception for external service errors."""
    pass


def _regex_reverse_search(
    regex: str,
    proc_response: subprocess.CompletedProcess,
) -> Optional[re.Match]:
    """
    Try to match the STDERR content with a regular expression from bottom to up.

    This is a complementary function for ``run_cmd``.

    :param str regex: The regular expression to try to match
    :param subprocess.CompletedProcess proc_response: the popen response to retrieve the STDERR from
    :return: the regex match or None if not matched
    :rtype: re.Match
    """
    # Start from the last log message since the failure occurs near the bottom
    for msg in reversed(proc_response.stderr.splitlines()):
        match = re.match(regex, msg)
        if match:
            return match
    return None


def run_cmd(
    cmd: List[str],
    params: Optional[Dict[str, Any]] = None,
    exc_msg: Optional[str] = None,
    strict: bool = True,
) -> str:
    """
    Run the given command with the provided parameters.

    :param list cmd: list of strings representing the command to be executed
    :param dict params: keyword parameters for command execution
    :param str exc_msg: an optional exception message when the command fails
    :param bool strict: when true function will throw exception when problem occurs
    :return: the command output
    :rtype: str
    :raises IIBError: if the command fails
    """
    exc_msg = exc_msg or 'An unexpected error occurred'
    if not params:
        params = {}
    params.setdefault('universal_newlines', True)
    params.setdefault('encoding', 'utf-8')
    params.setdefault('stderr', subprocess.PIPE)
    params.setdefault('stdout', subprocess.PIPE)

    logger.debug('Running the command "%s"', ' '.join(cmd))
    response: subprocess.CompletedProcess = subprocess.run(cmd, **params)

    if strict and response.returncode != 0:
        if set(['buildah', 'manifest', 'rm']) <= set(cmd) and 'image not known' in response.stderr:
            raise IIBError('Manifest list not found locally.')
        logger.error('The command "%s" failed with: %s', ' '.join(cmd), response.stderr)
        regex: str
        match: Optional[re.Match]
        if Path(cmd[0]).stem.startswith('opm'):
            # Capture the error message right before the help display
            regex = r'^(?:Error: )(.+)$'
            match = _regex_reverse_search(regex, response)
            if match:
                raise IIBError(f'{exc_msg.rstrip(".")}: {match.groups()[0]}')
            elif (
                '"permissive mode disabled" error="error deleting packages from'
                ' database: error removing operator package' in response.stderr
            ):
                raise IIBError("Error deleting packages from database")
        elif cmd[0] == 'buildah':
            # Check for HTTP 403 or 50X errors on buildah
            network_regexes = [
                r'.*([e,E]rror:? creating build container).*(:?(403|50[0-9]|125)\s?.*$)',
                r'.*(read\/write on closed pipe.*$)',
            ]
            for regex in network_regexes:
                match = _regex_reverse_search(regex, response)
                if match:
                    raise ExternalServiceError(f'{exc_msg}: {": ".join(match.groups()).strip()}')

        raise IIBError(exc_msg)

    return response.stdout


@dataclass
class BuildConfig:
    """Configuration for the multi-arch build."""
    image_name: str
    dockerfile_path: str
    context_path: str
    platforms: List[str]
    labels: List[str]
    cache_dir: str
    commit_sha: str = ""
    opm_version: str = "v1.40.0"  # Default OPM version
    retry_attempts: int = 3
    retry_delay: int = 5
    # Architecture mapping for platform names to expected architecture values
    arch_map: Dict[str, str] = field(default_factory=lambda: {
        'amd64': 'amd64',
        'arm64': 'arm64', 
        'ppc64le': 'ppc64le',
        's390x': 's390x'
    })


def generate_cache_locally(
    base_dir: str,
    fbc_dir: str,
    local_cache_path: str,
) -> None:
    """
    Generate the cache for the index image locally before building it.

    :param str base_dir: base directory where cache should be created.
    :param str fbc_dir: directory containing file-based catalog (JSON or YAML files).
    :param str local_cache_path: path to the locally generated cache.
    :return: Returns path to generated cache
    :rtype: str
    :raises: IIBError when cache was not generated

    """
    opm_version = os.environ.get('OPM_VERSION', 'v1.40.0')
    opm_binary = f"/usr/bin/opm-{opm_version}"

    cmd = [
        opm_binary,
        'serve',
        os.path.abspath(fbc_dir),
        f'--cache-dir={local_cache_path}',
        '--cache-only',
        '--termination-log',
        '/dev/null',
    ]

    logger.info('Generating cache for the file-based catalog')
    if os.path.exists(local_cache_path):
        shutil.rmtree(local_cache_path)
    run_cmd(cmd, {'cwd': base_dir}, exc_msg='Failed to generate cache for file-based catalog')

    # Check if the opm command generated cache successfully
    if not os.path.isdir(local_cache_path):
        error_msg = f"Cannot find generated cache at {local_cache_path}"
        logger.error(error_msg)
        raise IIBError(error_msg)


class MultiArchBuilder:
    """Orchestrates multi-architecture container builds."""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        
    def validate_dockerfile(self) -> bool:
        """Validate that Dockerfile exists."""
        if not Path(self.config.dockerfile_path).exists():
            logger.error(f"✗ Dockerfile not found: {self.config.dockerfile_path}")
            return False
        logger.info(f"✓ Dockerfile found: {self.config.dockerfile_path}")
        return True

    def _update_ca_trust(self, ca_bundle_path: str) -> None:
        """Update CA trust certificates."""
        if not Path(ca_bundle_path).exists():
            logger.warning(f"CA bundle not found at {ca_bundle_path}")
            return
            
        logger.info("Updating CA trust certificates")
        
        try:
            # Copy CA bundle to anchors directory
            run_cmd(['cp', '-vf', ca_bundle_path, '/etc/pki/ca-trust/source/anchors/'])
            
            # Update CA trust
            run_cmd(['update-ca-trust'])
            logger.info("✓ CA trust updated successfully")
        except IIBError as e:
            logger.error(f"✗ Failed to update CA trust: {e}")
            raise

    def _prepare_system(self) -> None:
        """Prepare the system for buildah operations."""
        logger.info("Preparing system for buildah operations")
        
        try:
            # Fix permissions on /var/lib/containers
            run_cmd(['chown', 'root:root', '/var/lib/containers'])
            
            # Configure short-name-mode
            run_cmd([
                'sed', '-i', 's/^\s*short-name-mode\s*=\s*.*/short-name-mode = "disabled"/',
                '/etc/containers/registries.conf'
            ])
            
            # Set up user namespace
            with open('/etc/subuid', 'a') as f:
                f.write('root:1:4294967294\n')
                
            logger.info("✓ System prepared successfully")
        except IIBError as e:
            logger.error(f"✗ Failed to prepare system: {e}")
            raise

    @retry(
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
        retry=retry_if_exception_type(ExternalServiceError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2),
    )
    def _build_image(self, arch: str, destination: str) -> None:
        """
        Build the index image for the specified architecture.

        :param str arch: the architecture to build this image for
        :param str destination: the destination image name
        :raises IIBError: if the build fails
        """
        logger.info(
            'Building the container image with the %s dockerfile for arch %s and tagging it as %s',
            os.path.basename(self.config.dockerfile_path),
            arch,
            destination,
        )
        
        # Prepare buildah command with improved options
        cmd = [
            'buildah',
            'bud',
            '--no-cache',
            '--format',
            'docker',
            '--override-arch',
            arch,
            '--arch',
            arch,
            '--tls-verify=true',
            '--ulimit', 'nofile=4096:4096',
            '-t',
            destination,
            '-f',
            self.config.dockerfile_path,
        ]
        
        # Add labels
        for label in self.config.labels:
            cmd.extend(['--label', label])
        

        # Add context
        cmd.append(self.config.context_path)
        
        # Execute build with retry logic
        run_cmd(cmd, {'timeout': 3600}, f"build for {arch} failed")
        
        # Verify architecture was set correctly
        logger.debug('Verifying that %s was built with expected arch %s', destination, arch)
        self._verify_image_architecture(destination, arch)

    def _verify_image_architecture(self, image_name: str, expected_arch: str) -> None:
        """
        Verify that the built image has the correct architecture using skopeo inspect.
        
        :param str image_name: the image name to verify
        :param str expected_arch: the expected architecture
        :raises ExternalServiceError: if architecture verification fails
        """
        try:
            # Get image architecture using skopeo inspect
            inspect_cmd = ['skopeo', 'inspect', '--no-tags', f'containers-storage:{image_name}']
            result = run_cmd(inspect_cmd, {'timeout': 60}, f"inspect {image_name} failed")
            image_data = json.loads(result)
            
            # Check architecture in image config
            arch = image_data.get('Architecture')
            
            if not arch:
                logger.warning(
                    'The "Architecture" was not found in image metadata. '
                    'Skipping the check that confirms if the architecture was set correctly.'
                )
                return
            
            # Map of platform names to expected architecture values
            # TODO: move to config
            arch_map = self.config.arch_map
            
            expected_arch_value = arch_map.get(expected_arch, expected_arch)
            
            if arch != expected_arch_value:
                logger.warning("Wrong arch created for %s", image_name)
                raise ExternalServiceError(
                    f'Wrong arch created, for image {image_name} '
                    f'expected arch {expected_arch_value}, found {arch}'
                )
            
            logger.info(f"✓ Architecture verification passed for {image_name}: {arch}")
            
        except Exception as e:
            if isinstance(e, ExternalServiceError):
                raise
            logger.warning(f"Could not verify architecture for {image_name}: {e}")
            # Don't fail the build if verification fails, just log a warning

    @retry(
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
        retry=retry_if_exception_type(IIBError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2),
    )
    def _create_and_push_manifest_list(
        self,
        arches: Set[str],
        platform_images: List[str],
    ) -> None:
        """
        Create and push the manifest list to the configured registry.

        :param set arches: a set of arches to create the manifest list for
        :param list platform_images: list of platform-specific image names
        :return: the pull specification of the manifest list
        :rtype: str
        :raises IIBError: if creating or pushing the manifest list fails
        """
        buildah_manifest_cmd = ['buildah', 'manifest']
        image_name_repo, image_name_tag = self.config.image_name.split(":", 1)
        # Initialize _tags with the output image tag
        _tags = [image_name_tag]
        if self.config.commit_sha:
            _tags.append(self.config.commit_sha)
        
        output_pull_specs = []
        for tag in _tags:
            output_pull_spec = f"{image_name_repo}:{tag}"
            output_pull_specs.append(output_pull_spec)
            try:
                run_cmd(
                    buildah_manifest_cmd + ['rm', output_pull_spec],
                    exc_msg=f'Failed to remove local manifest list. {output_pull_spec} does not exist',
                )
            except IIBError as e:
                error_msg = str(e)
                if 'Manifest list not found locally.' not in error_msg:
                    raise IIBError(f'Error removing local manifest list: {error_msg}')
                logger.debug(
                    'Manifest list cannot be removed. No manifest list %s found', output_pull_spec
                )
            logger.info('Creating the manifest list %s locally', output_pull_spec)
            run_cmd(
                buildah_manifest_cmd + ['create', output_pull_spec],
                exc_msg=f'Failed to create the manifest list locally: {output_pull_spec}',
            )
            for arch_image in platform_images:
                run_cmd(
                    buildah_manifest_cmd + ['add', output_pull_spec, arch_image],
                    exc_msg=(
                        f'Failed to add {arch_image} to the'
                        f' local manifest list: {output_pull_spec}'
                    ),
                )

            logger.debug('Pushing manifest list %s', output_pull_spec)
            run_cmd(
                buildah_manifest_cmd
                + [
                    'push',
                    '--all',
                    '--format',
                    'v2s2',
                    '--tls-verify=true',
                    output_pull_spec,
                    f'docker://{output_pull_spec}',
                ],
                exc_msg=f'Failed to push the manifest list to {output_pull_spec}',
            )
    
    def build_all(self, ca_bundle_path: Optional[str] = None) -> Dict[str, Any]:
        """Build multi-arch image and return results."""
        logger.info("Starting multi-architecture build")
        
        # Validate Dockerfile exists
        if not self.validate_dockerfile():
            raise RuntimeError("Dockerfile validation failed")
        
        # Update CA trust if bundle provided
        if ca_bundle_path:
            self._update_ca_trust(ca_bundle_path)
        
        # Prepare system
        self._prepare_system()
        
        # Generate cache using OPM
        logger.info("Generating cache using OPM")
        catalog_dir = Path(self.config.context_path) / "catalog"
        if not catalog_dir.exists():
            raise IIBError(f"Catalog directory not found at {catalog_dir}")
        
        logger.info(f"Found catalog directory at {catalog_dir}")
        generate_cache_locally(
            base_dir=self.config.context_path,
            fbc_dir=str(catalog_dir),
            local_cache_path=self.config.cache_dir
        )
        
        # Build images for each platform
        platform_images = []
        for platform in self.config.platforms:
            try:
                platform_clean = platform.strip()
                # output-image:tag-platform
                platform_image = f"{self.config.image_name}-{platform_clean}"
                self._build_image(platform_clean, platform_image)
                platform_images.append(platform_image)
            except Exception as e:
                logger.error(f"Failed to build for {platform}: {e}")
                raise
        
        # Create and push manifest
        logger.info("Creating and pushing multi-arch manifest")
        
        # Convert platforms to arch set
        arches = set(self.config.platforms)
        
        # Create and push manifest list
        self._create_and_push_manifest_list(arches, platform_images)
        
        # Get manifest digest
        inspect_cmd = ['skopeo', 'inspect', '--no-tags', self.config.image_name]
        result = run_cmd(inspect_cmd, {'timeout': 60}, "inspect manifest failed")
        manifest_data = json.loads(result)
        digest = manifest_data.get('Digest', '')
        
        return {
            'image_name': self.config.image_name,
            'digest': digest,
            'platforms': self.config.platforms,
            'platform_images': platform_images,
            'opm_version': self.config.opm_version
        }


def load_config_from_env() -> BuildConfig:
    """Load configuration from environment variables."""
    # Source code is extracted to /var/workdir/source by the use-trusted-artifact step
    source_dir = "/var/workdir/source"
    
    # If CONTEXT is relative, make it relative to source_dir
    context_path = os.environ.get('CONTEXT', '.')
    if not context_path.startswith('/'):
        context_path = os.path.join(source_dir, context_path)
    
    # If DOCKERFILE is relative, make it relative to source_dir
    dockerfile_path = os.environ.get('DOCKERFILE', './Dockerfile')
    if not dockerfile_path.startswith('/'):
        dockerfile_path = os.path.join(source_dir, dockerfile_path)
    
    return BuildConfig(
        image_name=os.environ.get('IMAGE', ''),
        dockerfile_path=dockerfile_path,
        context_path=context_path,
        platforms=os.environ.get('PLATFORMS', 'amd64,arm64,ppc64le,s390x').split(','),
        labels=os.environ.get('LABELS', '').split(',') if os.environ.get('LABELS') else [],
        cache_dir=os.environ.get('CACHE_DIR', '/var/workdir/cache'),
        commit_sha=os.environ.get('COMMIT_SHA', ''),
        opm_version=os.environ.get('OPM_VERSION', 'v1.40.0'),
        retry_attempts=int(os.environ.get('RETRY_ATTEMPTS', '3')),
        retry_delay=int(os.environ.get('RETRY_DELAY', '5'))
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multi-architecture container builder')
    parser.add_argument('--ca-bundle', help='Path to CA bundle file')
    parser.add_argument('--output', help='Path to output results JSON file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration from environment
        config = load_config_from_env()
        
        # Validate required fields
        if not config.image_name or not config.commit_sha:
            raise ValueError("IMAGE name and COMMIT_SHA are required")
        
        # Create builder and run build
        builder = MultiArchBuilder(config)
        results = builder.build_all(args.ca_bundle)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        logger.info("Multi-architecture build completed successfully")
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 