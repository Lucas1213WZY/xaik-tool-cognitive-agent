const RANGE_URL = "../assets/demo/coax_cognitive_parameter_ranges.json";
const API_BASE = "";  // same origin when served by Flask; empty = relative

const STRATEGY_FOR_XAI = {
  Importance: [
    "Sensitive-features categorization",
    "Salient-features categorization",
    "Importance categorization",
  ],
  Attribution: ["Sensitive-features categorization", "Attribution Sum"],
  None: ["Sensitive-features categorization"],
};

const CONDITION_COLORS = {
  "w/ XAI":  { bg: "#006d77cc", border: "#006d77" },
  "w/o XAI": { bg: "#e29578cc", border: "#e29578" },
};

const HUMAN_COLORS = {
  "w/ XAI":  { bg: "#f4a26188", border: "#f4a261" },
  "w/o XAI": { bg: "#e9c46a88", border: "#e9c46a" },
};

const state = {
  ranges: null,
  scope: "xai",
  appId: "",
  xaiType: "",
  inputMode: "ids",
  values: {},
  simMode: "custom",
  selectedStrategies: new Set(),
  maxParticipants: 5,
  lastSimResult: null,
};

const els = {
  status: document.querySelector("#status"),
  appId: document.querySelector("#appId"),
  xaiType: document.querySelector("#xaiType"),
  rangeScope: document.querySelector("#rangeScope"),
  sliders: document.querySelector("#sliders"),
  instanceIds: document.querySelector("#instanceIds"),
  instancesJson: document.querySelector("#instancesJson"),
  humanResponses: document.querySelector("#humanResponses"),
  humanInstanceColumn: document.querySelector("#humanInstanceColumn"),
  humanResponseColumn: document.querySelector("#humanResponseColumn"),
  humanParticipantColumn: document.querySelector("#humanParticipantColumn"),
  payloadPreview: document.querySelector("#payloadPreview"),
  humanPreview: document.querySelector("#humanPreview"),
  resetSliders: document.querySelector("#resetSliders"),
  copyPayload: document.querySelector("#copyPayload"),
  downloadConfig: document.querySelector("#downloadConfig"),
  tabs: document.querySelectorAll(".layout .tab"),
  idsInput: document.querySelector("#idsInput"),
  jsonInput: document.querySelector("#jsonInput"),
  // simulation
  simMode: document.querySelector("#simMode"),
  maxParticipantsField: document.querySelector("#maxParticipantsField"),
  maxParticipants: document.querySelector("#maxParticipants"),
  strategyChecks: document.querySelector("#strategyChecks"),
  runSim: document.querySelector("#runSim"),
  runSimLabel: document.querySelector("#runSimLabel"),
  runSimSpinner: document.querySelector("#runSimSpinner"),
  resultsSection: document.querySelector("#resultsSection"),
  simSummaryBar: document.querySelector("#simSummaryBar"),
  chartView: document.querySelector("#chartView"),
  tableView: document.querySelector("#tableView"),
  humanNoteSpan: document.querySelector("#humanNoteSpan"),
  resultTabs: document.querySelectorAll("[data-result-view]"),
};

let accuracyChartInstance = null;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("is-error", isError);
}

function formatNumber(value) {
  if (Number.isInteger(value)) return String(value);
  return Number.parseFloat(value.toPrecision(6)).toString();
}

function parseInstanceIds(text) {
  return text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter(Number.isFinite);
}

function parseRealInstances(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    return { error: "Instances JSON is not valid JSON yet." };
  }
}

function parseCsvRows(text) {
  const rows = [];
  let row = [], cell = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i], nx = text[i + 1];
    if (ch === '"' && inQuotes && nx === '"') { cell += '"'; i++; }
    else if (ch === '"') { inQuotes = !inQuotes; }
    else if (ch === ',' && !inQuotes) { row.push(cell); cell = ""; }
    else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && nx === '\n') i++;
      row.push(cell);
      if (row.some((v) => v.trim() !== '')) rows.push(row);
      row = []; cell = "";
    } else { cell += ch; }
  }
  row.push(cell);
  if (row.some((v) => v.trim() !== '')) rows.push(row);
  if (!rows.length) return [];
  const headers = rows[0].map((h) => h.trim());
  return rows.slice(1).map((vals) =>
    Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]))
  );
}

function parseHumanRows(text) {
  const t = text.trim();
  if (!t) return [];
  if (t.startsWith("[") || t.startsWith("{")) {
    try {
      const p = JSON.parse(t);
      return Array.isArray(p) ? p : [p];
    } catch { return { error: "Human responses JSON is not valid JSON." }; }
  }
  return parseCsvRows(t);
}

function humanColumns(rows) {
  if (!Array.isArray(rows) || !rows.length || typeof rows[0] !== "object") return [];
  return [...new Set(rows.flatMap((r) => Object.keys(r)))];
}

function chooseColumn(columns, preferredNames) {
  const norm = new Map(columns.map((c) => [c.toLowerCase().replace(/\s+/g, "_"), c]));
  for (const name of preferredNames) {
    const match = norm.get(name.toLowerCase().replace(/\s+/g, "_"));
    if (match) return match;
  }
  return columns[0] || "";
}

function renderColumnOptions(select, columns, selected, includeNone = false) {
  select.innerHTML = "";
  if (includeNone) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "None";
    select.appendChild(o);
  }
  for (const col of columns) {
    const o = document.createElement("option");
    o.value = col; o.textContent = col;
    select.appendChild(o);
  }
  select.value = (columns.includes(selected) || selected === "") ? selected : (select.options[0]?.value || "");
}

function mappedHumanResponses() {
  const rows = parseHumanRows(els.humanResponses.value);
  if (!Array.isArray(rows)) return rows;
  const ic = els.humanInstanceColumn.value;
  const rc = els.humanResponseColumn.value;
  const pc = els.humanParticipantColumn.value;
  if (!rows.length || !ic || !rc) return [];
  return rows
    .map((row, index) => ({
      row_index: index,
      instance_id: Number(row[ic]),
      human_response: row[rc],
      participant_id: pc ? row[pc] : "",
      raw: row,
    }))
    .filter((r) => Number.isFinite(r.instance_id) && r.human_response !== undefined && r.human_response !== "");
}

// ---------------------------------------------------------------------------
// Ranges & sliders
// ---------------------------------------------------------------------------

function activeRanges() {
  if (!state.ranges) return {};
  if (state.scope === "xai") return state.ranges.by_app_xai?.[state.appId]?.[state.xaiType] || {};
  if (state.scope === "global") return state.ranges.global || {};
  const appR = state.ranges.by_app?.[state.appId] || {};
  return Object.keys(appR).length ? appR : state.ranges.global || {};
}

function defaultValues(ranges) {
  return Object.fromEntries(Object.entries(ranges).map(([n, s]) => [n, s.default]));
}

function coerceValue(raw, spec) {
  const v = Number(raw);
  if (!Number.isFinite(v)) return spec.default;
  const clamped = Math.min(Math.max(v, spec.min), spec.max);
  return spec.type === "integer" ? Math.round(clamped) : clamped;
}

function renderAppOptions() {
  els.appId.innerHTML = "";
  for (const id of (state.ranges.app_ids || [])) {
    const o = document.createElement("option");
    o.value = id; o.textContent = id;
    els.appId.appendChild(o);
  }
  state.appId = state.ranges.app_ids?.[0] || "";
  els.appId.value = state.appId;
}

function preferredXaiType() {
  const appXai = Object.keys(state.ranges.by_app_xai?.[state.appId] || {});
  return appXai[0] || state.ranges.xai_types?.[0] || "";
}

function renderXaiOptions(useAppDefault = false) {
  els.xaiType.innerHTML = "";
  for (const xt of (state.ranges.xai_types || [])) {
    const o = document.createElement("option");
    o.value = xt; o.textContent = xt;
    els.xaiType.appendChild(o);
  }
  if (useAppDefault || !(state.ranges.xai_types || []).includes(state.xaiType)) {
    state.xaiType = preferredXaiType();
  }
  els.xaiType.value = state.xaiType;
}

function renderSliders(reset = false) {
  const ranges = activeRanges();
  if (reset || !Object.keys(state.values).length) state.values = defaultValues(ranges);

  els.sliders.innerHTML = "";
  if (!Object.keys(ranges).length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = `No fitted CoAX parameter rows found for appId "${state.appId}" with XAI type "${state.xaiType}".`;
    els.sliders.appendChild(empty);
    renderPayload();
    return;
  }

  for (const [name, spec] of Object.entries(ranges)) {
    if (!(name in state.values)) state.values[name] = spec.default;
    state.values[name] = coerceValue(state.values[name], spec);

    const row = document.createElement("div");
    row.className = "slider-row";

    const label = document.createElement("div");
    label.className = "slider-name";
    label.innerHTML = `<strong>${name}</strong><span>${formatNumber(spec.min)} to ${formatNumber(spec.max)} · ${spec.type}</span>`;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = spec.min; slider.max = spec.max; slider.step = spec.step;
    slider.value = state.values[name];
    slider.setAttribute("aria-label", name);

    const valueInput = document.createElement("input");
    valueInput.className = "value-input";
    valueInput.type = "number";
    valueInput.min = spec.min; valueInput.max = spec.max; valueInput.step = spec.step;
    valueInput.value = state.values[name];
    valueInput.setAttribute("aria-label", `${name} value`);

    function update(next) {
      const v = coerceValue(next, spec);
      state.values[name] = v;
      slider.value = v;
      valueInput.value = v;
      renderPayload();
    }

    slider.addEventListener("input", (e) => update(e.target.value));
    valueInput.addEventListener("input", (e) => update(e.target.value));

    row.append(label, slider, valueInput);
    els.sliders.appendChild(row);
  }
  renderPayload();
}

// ---------------------------------------------------------------------------
// Strategy checkboxes (for simulation)
// ---------------------------------------------------------------------------

function availableStrategies() {
  const xai = state.xaiType?.charAt(0).toUpperCase() + state.xaiType?.slice(1).toLowerCase();
  return STRATEGY_FOR_XAI[xai] || Object.keys({
    "Sensitive-features categorization": 1,
    "Salient-features categorization": 1,
    "Importance categorization": 1,
    "Attribution Sum": 1,
  });
}

function renderStrategyCheckboxes() {
  const strategies = availableStrategies();
  els.strategyChecks.innerHTML = "";

  // Seed selected set with all available
  if (state.selectedStrategies.size === 0) {
    strategies.forEach((s) => state.selectedStrategies.add(s));
  } else {
    // Keep only those still valid for the current xai type
    for (const s of [...state.selectedStrategies]) {
      if (!strategies.includes(s)) state.selectedStrategies.delete(s);
    }
    strategies.forEach((s) => state.selectedStrategies.add(s));
  }

  for (const strategy of strategies) {
    const label = document.createElement("label");
    label.className = "check-label";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = strategy;
    cb.checked = state.selectedStrategies.has(strategy);
    cb.addEventListener("change", () => {
      if (cb.checked) state.selectedStrategies.add(strategy);
      else state.selectedStrategies.delete(strategy);
    });

    label.append(cb, document.createTextNode(" " + strategy));
    els.strategyChecks.appendChild(label);
  }
}

// ---------------------------------------------------------------------------
// Payload preview
// ---------------------------------------------------------------------------

function payload() {
  const instances =
    state.inputMode === "ids"
      ? { mode: "instance_ids", instance_ids: parseInstanceIds(els.instanceIds.value) }
      : { mode: "real_instances", instances: parseRealInstances(els.instancesJson.value) };
  return {
    appId: state.appId,
    xaiType: state.xaiType,
    cognitive_parameters: state.values,
    range_scope: state.scope,
    source_ranges: RANGE_URL,
    input: instances,
    human_response_mapping: {
      join_key: "instance_id",
      instance_column: els.humanInstanceColumn.value,
      response_column: els.humanResponseColumn.value,
      participant_column: els.humanParticipantColumn.value,
      rows: mappedHumanResponses(),
    },
  };
}

function renderPayload() {
  els.payloadPreview.textContent = JSON.stringify(payload(), null, 2);
}

function renderHumanMapping(resetColumns = false) {
  const rows = parseHumanRows(els.humanResponses.value);
  if (!Array.isArray(rows)) {
    els.humanPreview.textContent = JSON.stringify(rows, null, 2);
    renderPayload();
    return;
  }
  const columns = humanColumns(rows);
  if (resetColumns) {
    renderColumnOptions(els.humanInstanceColumn, columns,
      chooseColumn(columns, ["instance_id", "instance_index", "instance id", "instance", "trial_index"]));
    renderColumnOptions(els.humanResponseColumn, columns,
      chooseColumn(columns, ["response", "human_response", "predicted", "choice"]));
    renderColumnOptions(els.humanParticipantColumn, columns,
      chooseColumn(columns, ["participant_id", "participant id", "participant"]), true);
  }
  const mapped = mappedHumanResponses();
  els.humanPreview.textContent = JSON.stringify({
    parsed_rows: rows.length,
    mapped_rows: Array.isArray(mapped) ? mapped.length : 0,
    join_key: "instance_id",
    rows: Array.isArray(mapped) ? mapped.slice(0, 20) : mapped,
  }, null, 2);
  renderPayload();
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

function humanAccuracyByInstanceId() {
  const mapped = mappedHumanResponses();
  if (!Array.isArray(mapped) || !mapped.length) return null;

  const byInstance = {};
  for (const r of mapped) {
    byInstance[r.instance_id] = r.human_response;
  }
  return byInstance;
}

async function runSimulation() {
  if ([...state.selectedStrategies].length === 0) {
    setStatus("Select at least one strategy to simulate.", true);
    return;
  }

  els.runSim.disabled = true;
  els.runSimLabel.textContent = "Running…";
  els.runSimSpinner.classList.remove("is-hidden");
  setStatus("Running simulation…");

  const body = {
    app_id: state.appId,
    xai_type: state.xaiType,
    seed: 7,
    mode: state.simMode,
    cognitive_parameters: state.values,
    strategies: [...state.selectedStrategies],
    max_participants: state.maxParticipants,
  };

  try {
    const resp = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    state.lastSimResult = data;
    renderSimResults(data);
    setStatus(`Simulation complete — ${data.total_rows} trial rows generated.`);
  } catch (err) {
    setStatus(`Simulation failed: ${err.message}`, true);
    console.error(err);
  } finally {
    els.runSim.disabled = false;
    els.runSimLabel.textContent = "Run Simulation";
    els.runSimSpinner.classList.add("is-hidden");
  }
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

function renderSimResults(data) {
  const { summary, trials, total_rows } = data;

  els.resultsSection.classList.remove("is-hidden");

  renderSummaryBar(summary, total_rows);
  renderAccuracyChart(summary);
  renderTrialTable(trials);
}

function renderSummaryBar(summary, totalRows) {
  const testRows = summary.filter((r) => r.trial_type === "Test");
  if (!testRows.length) { els.simSummaryBar.innerHTML = ""; return; }

  const overall = testRows.reduce((acc, r) => {
    acc.correct += r.n_correct;
    acc.total += r.n_trials;
    return acc;
  }, { correct: 0, total: 0 });

  const pct = overall.total > 0 ? Math.round(100 * overall.correct / overall.total) : 0;
  const strategies = [...new Set(testRows.map((r) => r.strategy))];

  els.simSummaryBar.innerHTML = `
    <span class="sim-chip">Overall test accuracy: <strong>${pct}%</strong></span>
    <span class="sim-chip">${strategies.length} strateg${strategies.length === 1 ? "y" : "ies"}</span>
    <span class="sim-chip">${totalRows} total trials</span>
  `;
}

function renderAccuracyChart(summary) {
  const testRows = summary.filter((r) => r.trial_type === "Test");
  const strategies = [...new Set(testRows.map((r) => r.strategy))];
  const conditions = ["w/ XAI", "w/o XAI"];

  // Build datasets: one per condition
  const datasets = conditions.map((cond) => {
    const col = CONDITION_COLORS[cond];
    return {
      label: `CoAX ${cond}`,
      data: strategies.map((strat) => {
        const row = testRows.find((r) => r.strategy === strat && r.condition === cond);
        return row ? Math.round(row.accuracy * 100) : null;
      }),
      backgroundColor: col.bg,
      borderColor: col.border,
      borderWidth: 1.5,
      borderRadius: 4,
    };
  });

  // Optionally overlay human accuracy if human responses are provided
  const humanMap = humanAccuracyByInstanceId();
  if (humanMap && state.lastSimResult?.trials?.length) {
    // Compute human accuracy per strategy × condition by joining on instance_id
    const humanRows = (state.lastSimResult.trials || []).filter(
      (t) => t.trialType === "Test" && humanMap[t["Instance Index"]] !== undefined
    );

    if (humanRows.length) {
      const humanDatasets = conditions.map((cond) => {
        const col = HUMAN_COLORS[cond];
        return {
          label: `Human ${cond}`,
          data: strategies.map((strat) => {
            const rows = humanRows.filter(
              (r) => r["Strategy"] === strat && r["Tested w/ XAI"] === cond
            );
            if (!rows.length) return null;
            const correct = rows.filter(
              (r) => String(humanMap[r["Instance Index"]]) === String(r["predicted"])
            ).length;
            return Math.round((correct / rows.length) * 100);
          }),
          backgroundColor: col.bg,
          borderColor: col.border,
          borderWidth: 1.5,
          borderRadius: 4,
        };
      });
      datasets.push(...humanDatasets);
      els.humanNoteSpan.textContent = "Orange bars: human responses overlaid from pasted data.";
    }
  } else {
    els.humanNoteSpan.textContent = "";
  }

  // Shorten strategy labels for the axis
  const shortLabels = strategies.map((s) =>
    s.replace("categorization", "cat.").replace("Sensitive-features", "Sensitive").replace("Salient-features", "Salient").replace("Importance", "Importance").replace("Attribution Sum", "Attr. Sum")
  );

  if (accuracyChartInstance) accuracyChartInstance.destroy();

  const ctx = document.querySelector("#accuracyChart").getContext("2d");
  accuracyChartInstance = new Chart(ctx, {
    type: "bar",
    data: { labels: shortLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y ?? "N/A"}%`,
          },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          title: { display: true, text: "Correct Label (%)" },
          ticks: { callback: (v) => `${v}%` },
        },
        x: {
          title: { display: true, text: "Reasoning Strategy" },
        },
      },
    },
  });
}

function renderTrialTable(trials) {
  if (!trials?.length) {
    document.querySelector("#trialTable thead").innerHTML = "";
    document.querySelector("#trialTable tbody").innerHTML = "<tr><td colspan='9' style='text-align:center;color:var(--muted)'>No trial data.</td></tr>";
    return;
  }

  const cols = [
    { key: "#", fn: (_, i) => i + 1 },
    { key: "Strategy", fn: (r) => r["Strategy"] },
    { key: "Condition", fn: (r) => r["Tested w/ XAI"] },
    { key: "Trial Type", fn: (r) => r["trialType"] },
    { key: "Instance ID", fn: (r) => r["Instance Index"] },
    { key: "CoAX Pred", fn: (r) => r["predicted"] },
    { key: "Correct", fn: (r) => r["Correct"] === 1 ? "✓" : "✗" },
    { key: "Prob Correct", fn: (r) => (r["Prob correct"] != null ? (r["Prob correct"] * 100).toFixed(1) + "%" : "—") },
  ];

  const thead = document.querySelector("#trialTable thead");
  thead.innerHTML = `<tr>${cols.map((c) => `<th>${c.key}</th>`).join("")}</tr>`;

  const tbody = document.querySelector("#trialTable tbody");
  tbody.innerHTML = "";

  for (const [i, row] of trials.entries()) {
    const tr = document.createElement("tr");
    tr.className = row["Correct"] === 1 ? "row-correct" : "row-wrong";
    for (const col of cols) {
      const td = document.createElement("td");
      td.textContent = col.fn(row, i) ?? "—";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

// ---------------------------------------------------------------------------
// File download / clipboard
// ---------------------------------------------------------------------------

function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function loadRanges() {
  try {
    const resp = await fetch(RANGE_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.ranges = await resp.json();
    renderAppOptions();
    renderXaiOptions(true);
    renderSliders(true);
    renderStrategyCheckboxes();
    setStatus(`Loaded ${state.ranges.parameter_columns.length} parameter ranges from ${state.ranges.source.csv}`);
  } catch (err) {
    setStatus(
      `Could not load ${RANGE_URL}. Run the generator, then serve via: conda run -n xaikit-coax python demo/server.py`,
      true
    );
    console.error(err);
  }
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

els.appId.addEventListener("change", (e) => {
  state.appId = e.target.value;
  renderXaiOptions(true);
  renderSliders(true);
  renderStrategyCheckboxes();
});

els.xaiType.addEventListener("change", (e) => {
  state.xaiType = e.target.value;
  state.selectedStrategies.clear();  // reset so all new strategies get selected
  renderSliders(true);
  renderStrategyCheckboxes();
});

els.rangeScope.addEventListener("change", (e) => { state.scope = e.target.value; renderSliders(true); });
els.resetSliders.addEventListener("click", () => renderSliders(true));
els.instanceIds.addEventListener("input", renderPayload);
els.instancesJson.addEventListener("input", renderPayload);
els.humanResponses.addEventListener("input", () => renderHumanMapping(true));
els.humanInstanceColumn.addEventListener("change", () => renderHumanMapping(false));
els.humanResponseColumn.addEventListener("change", () => renderHumanMapping(false));
els.humanParticipantColumn.addEventListener("change", () => renderHumanMapping(false));

els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    state.inputMode = tab.dataset.mode;
    els.tabs.forEach((t) => t.classList.toggle("is-active", t === tab));
    els.idsInput.classList.toggle("is-hidden", state.inputMode !== "ids");
    els.jsonInput.classList.toggle("is-hidden", state.inputMode !== "json");
    renderPayload();
  });
});

els.copyPayload.addEventListener("click", async () => {
  await navigator.clipboard.writeText(els.payloadPreview.textContent);
});

els.downloadConfig.addEventListener("click", () => {
  downloadJSON("coax_demo_config.json", payload());
});

els.simMode.addEventListener("change", (e) => {
  state.simMode = e.target.value;
  els.maxParticipantsField.classList.toggle("is-hidden", state.simMode !== "from_csv");
});

els.maxParticipants.addEventListener("change", (e) => {
  state.maxParticipants = Math.max(1, parseInt(e.target.value, 10) || 5);
});

els.runSim.addEventListener("click", runSimulation);

els.resultTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    els.resultTabs.forEach((t) => t.classList.toggle("is-active", t === tab));
    const view = tab.dataset.resultView;
    els.chartView.classList.toggle("is-hidden", view !== "chart");
    els.tableView.classList.toggle("is-hidden", view !== "table");
  });
});

loadRanges();
