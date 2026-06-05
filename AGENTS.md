# iib-tekton-utils

Tekton task for building multi-architecture Operator Index Images (IIB) using Python orchestration with buildah.

## What This Does

Builds container images for OLM file-based catalogs across multiple architectures (amd64, arm64, ppc64le, s390x), generates OPM cache, and pushes multi-arch manifest lists.

## Tech Stack

- Python 3 (orchestration)
- Tekton Tasks (Kubernetes CI/CD)
- Buildah (container builds)
- OPM (Operator Package Manager)
- GitHub Actions (CI)

## Key Files

- `task/iib-image-builder-oci-ta/iib-image-builder-oci-ta.yaml` - Main Tekton Task definition
- `task/iib-image-builder-oci-ta/multi-arch-builder.py` - Python build orchestrator
- `Containerfile.iib-build-task` - Container image for the build task
- `task/iib-image-builder-oci-ta/README.md` - Task parameters and usage

## Directory Structure

```
task/iib-image-builder-oci-ta/   # Tekton task + Python script
.github/workflows/               # CI workflows (build-and-push, pr-ci)
```

## Commands

```bash
# Build the container image locally
buildah bud -f Containerfile.iib-build-task -t iib-build-task .

# Lint Python
python3 -m py_compile task/iib-image-builder-oci-ta/multi-arch-builder.py
```

## Architecture Notes

- `MultiArchBuilder` class in multi-arch-builder.py:201 orchestrates the build
- `generate_cache_locally()` at :144 runs OPM to create FBC cache
- Retry logic via tenacity for buildah operations (:271, :365)
- Exception hierarchy: `IIBBaseException` > `IIBError`, `ExternalServiceError`

## Important Patterns

- Dockerfile must exist at configured path (validated at :207)
- Cache is generated before build, copied into build context (:459-480)
- Each platform builds separately, then merged into manifest list (:483-499)
- Results output as JSON with digest, platforms, image ref

## Related Docs

- [OLM File-Based Catalogs](https://olm.operatorframework.io/docs/concepts/olm-architecture/operator-catalog/creating-a-catalog/#file-based-catalogs)
- [OPM](https://github.com/operator-framework/operator-registry)
- [Tekton Tasks](https://tekton.dev/docs/pipelines/tasks/)
