const RANGE_URL = "../assets/demo/coax_cognitive_parameter_ranges.json";

const state = {
  ranges: null,
  scope: "xai",
  appId: "",
  xaiType: "",
  inputMode: "ids",
  values: {},
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
  tabs: document.querySelectorAll(".tab"),
  idsInput: document.querySelector("#idsInput"),
  jsonInput: document.querySelector("#jsonInput"),
};

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
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
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
  let row = [];
  let cell = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && inQuotes && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  row.push(cell);
  if (row.some((value) => value.trim() !== "")) rows.push(row);
  if (!rows.length) return [];

  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))
  );
}

function parseHumanRows(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    try {
      const parsed = JSON.parse(trimmed);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      return { error: "Human responses JSON is not valid JSON." };
    }
  }
  return parseCsvRows(trimmed);
}

function humanColumns(rows) {
  if (!Array.isArray(rows) || !rows.length || typeof rows[0] !== "object") return [];
  return [...new Set(rows.flatMap((row) => Object.keys(row)))];
}

function chooseColumn(columns, preferredNames) {
  const normalized = new Map(columns.map((column) => [column.toLowerCase().replace(/\s+/g, "_"), column]));
  for (const name of preferredNames) {
    const match = normalized.get(name.toLowerCase().replace(/\s+/g, "_"));
    if (match) return match;
  }
  return columns[0] || "";
}

function renderColumnOptions(select, columns, selected, includeNone = false) {
  select.innerHTML = "";
  if (includeNone) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "None";
    select.appendChild(option);
  }
  for (const column of columns) {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    select.appendChild(option);
  }
  select.value = columns.includes(selected) || selected === "" ? selected : select.options[0]?.value || "";
}

function mappedHumanResponses() {
  const rows = parseHumanRows(els.humanResponses.value);
  if (!Array.isArray(rows)) return rows;
  const instanceColumn = els.humanInstanceColumn.value;
  const responseColumn = els.humanResponseColumn.value;
  const participantColumn = els.humanParticipantColumn.value;

  if (!rows.length || !instanceColumn || !responseColumn) return [];

  return rows
    .map((row, index) => ({
      row_index: index,
      instance_id: Number(row[instanceColumn]),
      human_response: row[responseColumn],
      participant_id: participantColumn ? row[participantColumn] : "",
      raw: row,
    }))
    .filter((row) => Number.isFinite(row.instance_id) && row.human_response !== undefined && row.human_response !== "");
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
    renderColumnOptions(
      els.humanInstanceColumn,
      columns,
      chooseColumn(columns, ["instance_id", "instance_index", "instance id", "instance", "trial_index"])
    );
    renderColumnOptions(
      els.humanResponseColumn,
      columns,
      chooseColumn(columns, ["response", "human_response", "predicted", "choice"])
    );
    renderColumnOptions(
      els.humanParticipantColumn,
      columns,
      chooseColumn(columns, ["participant_id", "participant id", "participant"]),
      true
    );
  }

  const mapped = mappedHumanResponses();
  els.humanPreview.textContent = JSON.stringify(
    {
      parsed_rows: rows.length,
      mapped_rows: Array.isArray(mapped) ? mapped.length : 0,
      join_key: "instance_id",
      rows: Array.isArray(mapped) ? mapped.slice(0, 20) : mapped,
    },
    null,
    2
  );
  renderPayload();
}

function activeRanges() {
  if (!state.ranges) return {};
  if (state.scope === "xai") {
    return state.ranges.by_app_xai?.[state.appId]?.[state.xaiType] || {};
  }
  if (state.scope === "global") return state.ranges.global || {};
  const appRanges = state.ranges.by_app?.[state.appId] || {};
  return Object.keys(appRanges).length ? appRanges : state.ranges.global || {};
}

function defaultValues(ranges) {
  return Object.fromEntries(
    Object.entries(ranges).map(([name, spec]) => [name, spec.default])
  );
}

function coerceValue(rawValue, spec) {
  const value = Number(rawValue);
  if (!Number.isFinite(value)) return spec.default;
  const clamped = Math.min(Math.max(value, spec.min), spec.max);
  return spec.type === "integer" ? Math.round(clamped) : clamped;
}

function renderAppOptions() {
  els.appId.innerHTML = "";
  const appIds = state.ranges.app_ids || [];
  for (const appId of appIds) {
    const option = document.createElement("option");
    option.value = appId;
    option.textContent = appId;
    els.appId.appendChild(option);
  }
  state.appId = appIds[0] || "";
  els.appId.value = state.appId;
}

function preferredXaiType() {
  const appXaiTypes = Object.keys(state.ranges.by_app_xai?.[state.appId] || {});
  return appXaiTypes[0] || state.ranges.xai_types?.[0] || "";
}

function renderXaiOptions(useAppDefault = false) {
  els.xaiType.innerHTML = "";
  const xaiTypes = state.ranges.xai_types || [];
  for (const xaiType of xaiTypes) {
    const option = document.createElement("option");
    option.value = xaiType;
    option.textContent = xaiType;
    els.xaiType.appendChild(option);
  }
  if (useAppDefault || !xaiTypes.includes(state.xaiType)) {
    state.xaiType = preferredXaiType();
  }
  els.xaiType.value = state.xaiType;
}

function renderSliders(reset = false) {
  const ranges = activeRanges();
  if (reset || !Object.keys(state.values).length) {
    state.values = defaultValues(ranges);
  }

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
    if (!(name in state.values)) {
      state.values[name] = spec.default;
    }
    state.values[name] = coerceValue(state.values[name], spec);

    const row = document.createElement("div");
    row.className = "slider-row";

    const label = document.createElement("div");
    label.className = "slider-name";
    label.innerHTML = `
      <strong>${name}</strong>
      <span>${formatNumber(spec.min)} to ${formatNumber(spec.max)} · ${spec.type}</span>
    `;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = spec.min;
    slider.max = spec.max;
    slider.step = spec.step;
    slider.value = state.values[name];
    slider.setAttribute("aria-label", name);

    const value = document.createElement("input");
    value.className = "value-input";
    value.type = "number";
    value.min = spec.min;
    value.max = spec.max;
    value.step = spec.step;
    value.value = state.values[name];
    value.setAttribute("aria-label", `${name} value`);

    function update(nextValue) {
      const coerced = coerceValue(nextValue, spec);
      state.values[name] = coerced;
      slider.value = coerced;
      value.value = coerced;
      renderPayload();
    }

    slider.addEventListener("input", (event) => update(event.target.value));
    value.addEventListener("input", (event) => update(event.target.value));

    row.append(label, slider, value);
    els.sliders.appendChild(row);
  }
  renderPayload();
}

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

function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2) + "\n"], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function loadRanges() {
  try {
    const response = await fetch(RANGE_URL);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.ranges = await response.json();
    renderAppOptions();
    renderXaiOptions(true);
    renderSliders(true);
    setStatus(
      `Loaded ${state.ranges.parameter_columns.length} parameter ranges from ${state.ranges.source.csv}`
    );
  } catch (error) {
    setStatus(
      `Could not load ${RANGE_URL}. Run the generator, then serve the repo root with python -m http.server 8000.`,
      true
    );
    console.error(error);
  }
}

els.appId.addEventListener("change", (event) => {
  state.appId = event.target.value;
  renderXaiOptions(true);
  renderSliders(true);
});

els.xaiType.addEventListener("change", (event) => {
  state.xaiType = event.target.value;
  renderSliders(true);
});

els.rangeScope.addEventListener("change", (event) => {
  state.scope = event.target.value;
  renderSliders(true);
});

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
    els.tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
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

loadRanges();
