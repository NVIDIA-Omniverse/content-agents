# Physics Agent Auto-Tuning

This guide covers the Physics Agent auto-tuning and refine surfaces: architecture,
extension points, config-driven CLI usage, prompt-driven CLI usage, examples, and
REST integration status.

## What Tune And Refine Do

`tune` starts from a simulation-ready USD that already has physics schemas from
`apply_physics`. It patches tunable physics parameters, evaluates each candidate
in a simulation backend, records trial history, writes the best parameter set,
and can ask a VLM judge to review the result.

`refine` wraps `tune` in an iterative loop. Each iteration runs a full tune pass,
asks the judge whether the result is good enough, and, when the answer is not
good enough, asks an LLM to rewrite the scenario YAML for the next iteration.

## Architecture

The tuning stack has five layers:

| Layer | Main code | Responsibility |
|-------|-----------|----------------|
| Interface | `physics_agent.cli`, `physics_agent.api`, `physics_agent_service.service.routers.tune_router` | CLI, Python API, and REST request surfaces |
| Orchestration | `physics_agent.tuning.runner`, `physics_agent.tasks.iterative_physics_refinement` | Load scenario or prompt, run optimizer trials, manage artifacts, drive refine loop |
| Scenario | `physics_agent.tuning.scenario`, `physics_agent.tuning.scenarios.*` | Parse scenario YAML, build simulation scenes, compute scenario metrics |
| Optimization | `physics_agent.tuning.optimizers` | Dispatch `auto`, `botorch`, `random`, or `cma-es` over scenario bounds |
| Simulation backend | `physics_agent.tuning.backend`, `physics_agent.tuning.ovphysx_backend`, `physics_agent.tuning.newton_backend` | Evaluate one candidate parameter set and return a scalar score |

The single-shot tune flow is:

```text
scenario YAML or user prompt
  -> TuneInput / POST /tune
  -> scenario loader or prompt interpreter
  -> optimizer trial loop
  -> patch physics USD
  -> simulation backend evaluates scenario
  -> best_params.json, history.jsonl, tuned_physics.usda, tune_results.json, report.md
  -> optional VLM judge and optional comparison.png
```

The iterative refine flow is:

```text
initial scenario + physics USD + user prompt
  -> RefineInput / physics-agent refine
  -> tune iteration
  -> VLM judge
  -> scenario_refine LLM rewrite when judge returns continue
  -> next tune iteration
  -> final/ snapshot and refine_summary.json
```

## Config-Driven CLI Usage

Install the optional tuning dependencies when using the production optimizer.
This extra covers BoTorch and tuning-side dependencies; it does not install
simulator-specific extras or the separate OvPhysX daemon environment:

```bash
uv pip install -e "apps/physics_agent[tuning]"
```

For `--engine newton` runs, install the Newton extra in the parent environment.
It includes `physics-agent[tuning]` and the PyPI
`newton[sim,importers]>=1.2.0,<2.0` dependency. Newton runs in-process, so no
daemon venv is required:

```bash
uv pip install -e "apps/physics_agent[newton]"
```

For production `--engine ovphysx` runs, bootstrap the daemon venv as well. The
daemon uses a separate Python environment because `ovphysx` bundles an OpenUSD
version that conflicts with the parent process:

```bash
export WU_OVPHYSX_VENV_DIR="${WU_OVPHYSX_VENV_DIR:-$HOME/.cache/wu/ovphysx_venv}"
uv venv "$WU_OVPHYSX_VENV_DIR"
VIRTUAL_ENV="$WU_OVPHYSX_VENV_DIR" uv pip install ovphysx \
  --extra-index-url https://pypi.nvidia.com
```

Run the normal pipeline first to produce a physics-authored USD:

```bash
physics-agent run apps/physics_agent/configs/lightbulb.yaml
```

Then tune that physics USD with a scenario YAML:

```bash
physics-agent tune apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd path/to/asset_physics.usda \
  --engine ovphysx \
  --optimizer auto \
  --output-dir output/tune
```

For Newton, choose a scenario whose parameters are supported by the Newton
MuJoCo path. The tire bounce reference uses `contact_ke` and `contact_kd`
instead of USD restitution:

```bash
physics-agent tune apps/physics_agent/configs/tuning/tire_b01_drop_settle_newton.yaml \
  --physics-usd path/to/tire_physics.usda \
  --engine newton \
  --optimizer random \
  --output-dir output/tire_tune_newton
```

`--optimizer auto` resolves to BoTorch. If BoTorch is not installed, it raises an
install-hint error instead of silently falling back to random search. The
`random` and `cma-es` optimizers are always available baselines.

Use `refine` when the judge should be allowed to rewrite the scenario and run
more tune iterations:

```bash
physics-agent refine apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd path/to/asset_physics.usda \
  --user-prompt "make it bouncy" \
  --output-dir output/refine \
  --engine ovphysx \
  --optimizer random \
  --max-trials 4 \
  --max-iterations 3 \
  --score-threshold 0.7
```

The same `tune` and `refine` surfaces accept `--engine newton` when the scenario
uses Newton-supported parameters. Newton supports `mass_scale`,
`dynamic_friction`, `contact_ke`, and `contact_kd`; it rejects
`static_friction` and `restitution` before queueing the run because the current
Newton importer and MuJoCo solver path cannot apply those USD-authored knobs
effectively. Use `--engine ovphysx` for static-friction or restitution tuning.

## Prompt-Driven CLI Usage

`tune` can run without a scenario YAML when `--user-prompt` and `--physics-usd`
are supplied. The prompt interpreter authors a scenario and stores the inferred
scenario as an audit artifact.

```bash
physics-agent tune \
  --user-prompt "make this tire bounce higher after it hits the ground" \
  --physics-usd path/to/tire_physics.usda \
  --engine ovphysx \
  --optimizer auto \
  --output-dir output/tire_tune
```

You can also provide both a scenario YAML and `--user-prompt`. In that mode,
explicit YAML fields win on conflicts and the prompt interpreter fills missing
fields.

Reference media can be attached to tune or refine judge calls:

```bash
physics-agent tune apps/physics_agent/configs/tuning/drop_settle.yaml \
  --physics-usd path/to/asset_physics.usda \
  --reference-image reference.png \
  --reference-video observed_motion.mp4 \
  --judge-max-tokens 2048 \
  --judge-temperature 0
```

## Scenario YAML

A scenario declares the scenario kind, target simulation settings, optional
judge settings, and tunable parameter bounds.

```yaml
name: drop_settle
metric: settle_distance

target:
  drop_height_m: 0.5
  duration_s: 2.0
  gravity: -9.81
  sample_fps: 30
  cameras: ["+x+y+z"]
  vlm_check: "off"
  record_video: "off"

judge:
  temperature: 0.0
  max_tokens: 2048

parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
  - name: static_friction
    min: 0.05
    max: 1.5
  - name: dynamic_friction
    min: 0.05
    max: 1.5
  - name: restitution
    min: 0.0
    max: 1.0
```

Reference configs:

| Config | Purpose |
|--------|---------|
| `apps/physics_agent/configs/tuning/drop_settle.yaml` | Generic drop-settle scenario and schema comments |
| `apps/physics_agent/configs/tuning/tire_b01_drop_settle.yaml` | Tire_B01 drop-settle scenario with camera ground bias and video recording |
| `apps/physics_agent/configs/tuning/tire_b01_drop_settle_newton.yaml` | Tire_B01 Newton drop-settle scenario using contact stiffness/damping for bounce tuning |
| `apps/physics_agent/configs/tire_bounce.yaml` | Public classification/apply config used to create a physics USD for tire bounce tuning |

## Extension Points

### Add A Scenario Kind

Scenario kinds are registered in `physics_agent.tuning.types.SUPPORTED_SCENARIOS`
and advertised per engine in
`physics_agent.tuning.scenarios.SUPPORTED_SCENARIOS_PER_ENGINE`. Add a module
under `physics_agent/tuning/scenarios/` that exports an `evaluate(...)` callable,
then add it to `physics_agent.tuning.scenarios.resolve(...)`.

The runner and REST router both use the same capability map, so unsupported
engine/scenario pairs fail before an expensive background job is queued.

### Add A Metric

Metrics for `drop_settle` live in
`physics_agent.tuning.scenarios.drop_settle._METRICS`. A metric receives a
`MetricContext` and returns a scalar where lower is better. Quantities that are
physically "higher is better" should be negated before returning.

### Add A Tunable Parameter

The supported tunable parameter keys live in
`physics_agent.tuning.types.SUPPORTED_PARAM_KEYS`, with fallback bounds in
`DEFAULT_PARAM_BOUNDS`. Add tests before expanding this set because existing
scenarios, prompt interpretation, USD patching, and report artifacts assume
these names.

### Add An Optimizer

Optimizers are dispatched through `physics_agent.tuning.optimizers`. Add the
public optimizer name to `SUPPORTED_OPTIMIZERS`, implement a runner that calls
the supplied `evaluate(params)` callback up to `max_trials`, and wire it through
`resolve_optimizer(...)` / `get_runner(...)`.

### Add A Simulation Backend

Backends implement `physics_agent.tuning.backend.TuningBackend.evaluate(...)`.
Register the engine name in `SUPPORTED_ENGINES` and return an instance from
`get_backend(...)`. Production backends should lazy-import heavy dependencies so
the base package can still import without optional tuning extras.

OvPhysX runs through an isolated daemon venv because its bundled OpenUSD can
conflict with the parent process. Newton is loaded from the parent environment
through `apps/physics_agent[newton]` and should expose its supported tunable
parameters through `physics_agent.tuning.capabilities` so CLI and REST callers
fail before an expensive simulation job is queued.

### Extend Judge Evidence

Reference images and videos are normalized in
`physics_agent.tuning.visual_evidence`. Judge outputs are persisted under
`judge.extra.visual_evidence` in `tune_results.json`, `judge_result.json`, and
`report.md`. Media-backed tune/refine paths fail closed when the judge cannot
produce a real verdict.

## REST Integration

The service exposes single-shot tuning through `/tune`:

| Endpoint | Purpose |
|----------|---------|
| `POST /tune` | Create a tune session from a physics USD upload, S3 URI, or completed pipeline `source_session_id` |
| `GET /tune/{session_id}/status` | Poll trial count, best score, best params, and terminal status |
| `GET /tune/{session_id}/results` | Fetch final or partial tune results plus artifact URLs |
| `GET /tune/{session_id}/events` | Stream trial progress over SSE on the executing instance |
| `POST /tune/{session_id}/cancel` | Cooperatively cancel a pending or running tune session |
| `GET /tune/{session_id}/artifacts/{name}` | Download `best_params.json`, `tune_results.json`, `history.jsonl`, `report.md`, `tuned_physics.usda`, or `comparison.png` |

The REST worker delegates to `physics_agent.tuning.arun_tune` through
`apps/physics_agent_service/service/workers/tune_executor.py` and reuses the
same `SessionManager`, `JobRegistry`, `EventBus`, cancellation marker, and
artifact-store sync patterns as `/pipeline`.

There is currently no first-class `/refine` REST route. Iterative refine is
available through the CLI and Python API (`RefineInput`, `run_refine`,
`arun_refine`). REST callers that need true multi-iteration refine need a future
route rather than relying on `/tune`'s `judge_max_iterations` field; that field
is preserved for compatibility and audit metadata, but single-shot `/tune` does
not re-run tuning when the judge returns `continue`.

## Service Client Status

The bundled Python service client currently focuses on `/pipeline` workflows.
For `/tune`, use raw HTTP, generated clients from `openapi.yaml`, or add a
client wrapper that mirrors the service API documented in
`apps/physics_agent_service/docs/api.md`.

## Release QA Checklist

For a release QA pass, verify at least one run from each interface:

| Surface | Minimum check |
|---------|---------------|
| Config CLI | `physics-agent tune apps/physics_agent/configs/tuning/drop_settle.yaml --physics-usd ...` |
| Prompt CLI | `physics-agent tune --user-prompt "make this object bouncy" --physics-usd ...` |
| Refine CLI | `physics-agent refine ... --user-prompt ... --max-iterations 2` |
| Python API | `run_tune(TuneInput(...))` and `run_refine(RefineInput(...))` construct and return typed outputs |
| REST tune | `POST /tune`, status polling, results, artifact download, and cancellation behavior |
| Service gap | Confirm `/refine` is documented as absent unless a future PR adds it |
