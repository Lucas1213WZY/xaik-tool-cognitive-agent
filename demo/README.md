# CoAX Interactive Cognitive Parameter Demo

This demo is intentionally separate from `src/`.

## 1. Generate Slider Ranges

Run this from the repository root:

```bash
conda run -n xaikit-coax python demo/generate_coax_parameter_ranges.py
```

It reads `assets/param_config/CoAX_cog_param.csv` and writes
`assets/demo/coax_cognitive_parameter_ranges.json`.

## 2. Start the Demo Server

```bash
conda run -n xaikit-coax python demo/server.py
```

Then open **http://localhost:5000/demo/**

The server handles both the static demo files and the `/api/simulate` endpoint,
so the "Run Simulation" button works end-to-end.

### Legacy static server (payload-only, no simulation)

If you only need to inspect the payload without running a live simulation:

```bash
python -m http.server 8000
# open http://localhost:8000/demo/
```

## What The UI Does

- **Cognitive Parameter sliders** — set `k`, `sensitivity`, `retrieval_threshold`,
  and (for Attribution type) `scaling_factor` using the fitted CSV ranges.
- **Run Simulation** — sends the slider values to the server, which calls
  `simulate_virtual_experiment` and returns trial rows.
  - *Custom mode*: runs a single virtual participant with exactly the slider values.
  - *From CSV mode*: runs the first N fitted participants from the CSV.
- **Accuracy Chart** — bar chart of test-trial accuracy by reasoning strategy and
  condition (w/ XAI vs w/o XAI). If human responses are pasted, they are overlaid
  as additional bars (matching the Fig. 9 style from the CoAX paper).
- **Trial Table** — scrollable row-level view of every simulated trial: strategy,
  condition, trial type, CoAX prediction, correctness, and probability.
- **Download Config / Copy** — export the current payload as JSON.
