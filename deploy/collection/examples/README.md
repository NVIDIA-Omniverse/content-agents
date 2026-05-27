# Collection Deployment Examples

These examples are endpoint contracts for common deployment shapes. Copy one to
`deploy/collection/.local.yaml` or pass it directly with `-c`.

Secrets stay in the repo-root `.env`; do not put keys in these YAML files.

## Examples

- `minimal.yaml`: CPU-only agents plus a local or external OVRTX endpoint.
- `brev-render-only.yaml`: CPU-only agents with Brev-hosted OVRTX.
- `brev-render-embeddings.yaml`: Brev-hosted OVRTX and embedding NIM.
- `full-brev.yaml`: target shape for all dependencies hosted behind Brev
  forwards.

## Commands

```bash
./deploy/collection/deploy.py -c deploy/collection/examples/minimal.yaml plan
./deploy/collection/deploy.py -c deploy/collection/examples/minimal.yaml up
./deploy/collection/deploy.py -c deploy/collection/examples/minimal.yaml smoke
```

For Brev-backed examples:

```bash
./deploy/collection/deploy.py -c deploy/collection/examples/full-brev.yaml brev-plan
./deploy/collection/deploy.py -c deploy/collection/examples/full-brev.yaml brev-provision --execute
```

For Docker-hosted agents calling Brev endpoints, bind SSH forwards to all host
interfaces and use `host.docker.internal` in the YAML.

For local GPU sidecars, use the standalone Compose files under
`deploy/collection/`: `docker-compose.vlm.yml`, `docker-compose.llm.yml`,
`docker-compose.embeddings.yml`, and `docker-compose.image-gen.yml`. On Brev
GPU hosts with a mounted data disk, point the model cache variables at writable
disk paths before startup, for example
`COLLECTION_IMAGE_GEN_CACHE_VOLUME=/opt/dlami/nvme/nim-cache`.
