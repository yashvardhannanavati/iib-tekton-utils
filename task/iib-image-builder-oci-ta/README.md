# iib-image-builder-oci-ta task

The iib-image-builder task builds source code into multi-architecture index images using Python orchestration with buildah. This task is specifically designed for building Operator Index Images that contain file-based catalogs.

The task performs the following operations:
- Generates OPM cache from file-based catalog (JSON or YAML files)
- Builds container images for multiple architectures (amd64, arm64, ppc64le, s390x)
- Creates and pushes multi-architecture manifest list

## Parameters

|name|description|default value|required|
|---|---|---|---|
|COMMIT_SHA|The image is built from this commit.|""|true|
|CONTEXT|Path to the directory to use as context.|.|false|
|DOCKERFILE|Path to the Dockerfile to build.|./Dockerfile|false|
|IMAGE|Reference of the image buildah will produce.||true|
|LABELS|Additional key=value labels that should be applied to the image (comma-separated)|""|false|
|SOURCE_ARTIFACT|The Trusted Artifact URI pointing to the artifact with the application source code.||true|
|STORAGE_DRIVER|Storage driver to configure for buildah|overlay|false|
|PLATFORMS|Comma-separated list of platforms to build for (e.g., amd64,arm64,ppc64le,s390x)|amd64,arm64,ppc64le,s390x|false|
|OPM_VERSION|OPM version to use for cache generation|v1.40.0|false|
|caTrustConfigMapKey|The name of the key in the ConfigMap that contains the CA bundle data.|ca-bundle.crt|false|
|caTrustConfigMapName|The name of the ConfigMap to read CA bundle data from.|trusted-ca|false|

## Results

|name|description|
|---|---|
|IMAGE_DIGEST|Digest of the multi-arch image manifest|
|IMAGE_REF|Image reference of the built multi-arch image (includes digest)|
|IMAGE_URL|Image repository and tag where the built image was pushed|

## Usage

### Basic Usage

```yaml
apiVersion: tekton.dev/v1
kind: TaskRun
metadata:
  name: build-multi-arch-index
spec:
  taskRef:
    name: iib-image-builder-oci-ta
  params:
    - name: IMAGE
      value: "quay.io/myorg/my-index:v1.0.0"
    - name: COMMIT_SHA
      value: "abc123def456"
    - name: SOURCE_ARTIFACT
      value: "oci://source-artifact"
```

### Advanced Usage with Custom Parameters

```yaml
apiVersion: tekton.dev/v1
kind: TaskRun
metadata:
  name: build-multi-arch-index-advanced
spec:
  taskRef:
    name: iib-image-builder-oci-ta
  params:
    - name: IMAGE
      value: "quay.io/myorg/my-index:v1.0.0"
    - name: COMMIT_SHA
      value: "abc123def456"
    - name: SOURCE_ARTIFACT
      value: "oci://source-artifact"
    - name: PLATFORMS
      value: "amd64,arm64"
    - name: LABELS
      value: "<label-name>=<label-value>, <label2-name>=<label2-value>"
    - name: OPM_VERSION
      value: "v1.40.0"
    - name: CONTEXT
      value: "./operator"
    - name: DOCKERFILE
      value: "./operator/Dockerfile"
```

## Related Documentation

- [Operator Lifecycle Manager (OLM)](https://olm.operatorframework.io/)
- [File-based Catalogs](https://olm.operatorframework.io/docs/concepts/olm-architecture/operator-catalog/creating-a-catalog/#file-based-catalogs)
- [OPM (Operator Package Manager)](https://github.com/operator-framework/operator-registry)
- [Buildah](https://buildah.io/)
