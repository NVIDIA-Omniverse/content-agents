# Material Agent Large-Scene Quickstart

This example is a tiny synthetic USD stage for exercising the public
large-scene Material Agent service API. The input is one composed USD stage with
a valid `defaultPrim`; it is not a ZIP of independent USD files.

## Run Against The Service

Start `apps/material_agent_service`, then submit the scene:

```bash
python -m apps.material_agent_service.client.client \
  --base-url http://localhost:8000 \
  --email user@example.com \
  --large-scene \
  --scene-workers 1 \
  --scene-no-render \
  --scene-fail-on-validation-error \
  apps/material_agent_service/examples/large_scene/warehouse.usda
```

Equivalent `curl` request:

```bash
curl -X POST http://localhost:8000/pipeline \
  -F "usd_file=@apps/material_agent_service/examples/large_scene/warehouse.usda" \
  -F "user_email=user@example.com" \
  -F "large_scene=true" \
  -F "scene_workers=1" \
  -F "scene_no_render=true" \
  -F "scene_fail_on_validation_error=true"
```

Poll the session ID returned by the service:

```bash
curl http://localhost:8000/pipeline/<session_id>/status
curl http://localhost:8000/pipeline/<session_id>/results
```

Useful large-scene artifacts:

```bash
curl -O http://localhost:8000/artifacts/<session_id>/output
curl -O http://localhost:8000/artifacts/<session_id>/scene-manifest
curl -O http://localhost:8000/artifacts/<session_id>/scene-predictions
curl -O http://localhost:8000/artifacts/<session_id>/scene-validation-report
```

For scenes with external dependencies, package the composed stage as USDZ or
make sure the referenced files are resolvable by the USD runtime used by the
service.
