# Validation Agent Examples

These examples show Validation Agent V1 on realistic Content Agents release
assets without requiring users to run Material Agent, Physics Agent, or Texture
Agent first. The visual and source-asset physics examples consume public
SimReady assets that you download locally; the behavior example uses compact
checked-in evidence so it can run immediately.

Run all commands from the repository root after installing the package:

```bash
uv pip install -e . -e apps/validation_agent
```

## Example 1: Electrician's Toolbox Visual Check

This visual/reference example validates a public SimReady electrician's
toolbox USD against its reference thumbnail with `render_valid` and
`look_right`.

Download the public source asset into the local examples fixture directory:

```bash
SIMREADY_ROOT=apps/validation_agent/examples/fixtures/simready/github
mkdir -p "$SIMREADY_ROOT"

if [ ! -d "$SIMREADY_ROOT/simready-foundation" ]; then
  git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/NVIDIA/simready-foundation.git \
    "$SIMREADY_ROOT/simready-foundation"
fi

git -C "$SIMREADY_ROOT/simready-foundation" sparse-checkout add \
  sample_content/common_assets/props_general/obs_electricians_large_tool_box_a01/simready_usd
```

Run a prompt-driven render preflight:

```bash
TOOLBOX_PROP=obs_electricians_large_tool_box_a01
TOOLBOX_ASSET="sample_content/common_assets/props_general/$TOOLBOX_PROP/simready_usd"
TOOLBOX_ROOT="$SIMREADY_ROOT/simready-foundation/$TOOLBOX_ASSET"

validation-agent validate \
  --task "Validate that the public SimReady electrician's toolbox renders successfully." \
  --template render_valid \
  --render-backend remote \
  --render-view corner \
  --image-width 512 \
  --image-height 512 \
  --output-dir apps/validation_agent/examples/runs/electricians_toolbox_visual \
  "$TOOLBOX_ROOT/sm_obs_electricians_large_tool_box_a01_01.usd"
```

Run the config-driven path for full visual/reference judging:

```bash
validation-agent run \
  apps/validation_agent/examples/configs/electricians_toolbox_visual.yaml
```

The config opts into canonical USD visual evidence, so the visual templates use
fresh runtime renders rather than stale caller-provided PNG bundles. Set
`RENDER_ENDPOINT` or `NVCF_RENDER_FUNCTION_ID` for runtime rendering, and set a
VLM key for the selected `look_right` backend.

## Example 2: Steel Scaffold Known-Negative Physics Audit

Public prebuilt SimReady PhysX assets are useful source-asset audits, but some
of them are intentionally known-negative for `physics_sane`. This example uses
the public steel rolling scaffold PhysX USD and expects the missing physics
scene and rigid body findings. A matching known-negative result is reported as
`warn`; changed issue codes stay blocking. When a known-negative audit fails
because the observed issue codes drift, inspect
`metadata.expected_result.reason` in `validation_result.json`.

Download the public scaffold asset:

```bash
uv pip install -U huggingface_hub

hf download --repo-type dataset nvidia/PhysicalAI-SimReady-Warehouse-01 \
  --include "Props/general/SM_SteelRollingScaffold_A01_01/*" \
  --local-dir apps/validation_agent/examples/fixtures/simready/hf
```

Run the config:

```bash
validation-agent run \
  apps/validation_agent/examples/configs/steel_scaffold_known_negative_physics.yaml
```

Use `--fail-on-warn` only when you want known-negative source-asset audits to
block a CI job.

## Example 3: Steel Scaffold Behavior Evidence (Hello-World)

Validation Agent does not run simulation or tune/refine loops in V1. It
consumes already-produced motion or Physics Agent refine artifacts and maps
them into stable `physical_behavior` status semantics.

This compact fixture represents an approved steel scaffold rollout/refine
summary. The USDA supplies time-sampled rollout evidence, and
`policy.physical_behavior_refine_summary_path` supplies the judge/refine
decision that makes the example pass:

```bash
validation-agent run \
  apps/validation_agent/examples/configs/steel_scaffold_behavior_refine_summary.yaml
```

The expected result is `pass` with `physical_behavior=passed`, because the
checked-in `refine_summary.json` records an approved final judge decision. For
real QA, replace the fixture paths with a production Physics Agent refine
output directory containing `refine_summary.json`, `iter_N/`, and `final/`
artifacts.

## Output Files

Every run writes these stable artifacts under the selected output directory:

- `validation_request.json`: the effective request after CLI overrides.
- `validation_plan.json`: resolved inputs and ordered template plan.
- `validation_result.json`: final verdict, per-template statuses, issues,
  metrics, evidence, and recommended action.

Verdicts are CI-oriented:

- `pass`: selected templates passed.
- `warn`: validation completed with non-blocking issues; exit code is `0`
  unless `--fail-on-warn` is set.
- `fail`: blocking validation issue; exit code is `1`.
- `needs_refinement`: the asset or behavior needs another iteration; exit code
  is `1`.
- `planned`: dry-run output; exit code is `0`.
