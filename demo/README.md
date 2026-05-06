# CoAX Interactive Cognitive Parameter Demo

This demo is intentionally separate from `src/`.

## 1. Generate Slider Ranges

Run this from the repository root:

```bash
python demo/generate_coax_parameter_ranges.py
```

It reads:

```text
assets/param_config/CoAX_cog_param.csv
```

and writes:

```text
assets/demo/coax_cognitive_parameter_ranges.json
```

The JSON contains global ranges and per-`appId` ranges for the CoAX cognitive
parameters found in the fitted CSV.

It also includes ranges nested by `appId` and `XAIType`. This matters because
the fitted CSV does not give every row both `sensitivity` and `scaling_factor`:
the available sliders should reflect the selected app and explanation type.

## 2. Open The UI

Serve the repo root so the browser can load the generated asset:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/demo/
```

## What The UI Does

- Select `appId` interactively.
- Select `XAIType` interactively.
- Use app+XAI-type, app-level, or global CSV ranges.
- Set each cognitive parameter with a slider and numeric input.
- Enter either a list of `instance_ids` or real data instances as JSON.
- Paste real human response rows as CSV or JSON and map their instance and
  response columns. The payload exposes these rows under
  `human_response_mapping.rows`, joined by `instance_id`.
- Copy or download the resulting demo payload.
