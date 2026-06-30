import {
  buildPromptRows,
  formatSeconds,
  notePosition,
  parsePromptCsv,
  serializePromptCsv,
} from "./schedule.mjs";

const laneCount = 5;
const defaultRows = buildPromptRows({ count: 20, intervalSeconds: 3 });
const state = {
  prompts: defaultRows,
  events: [],
  mode: "ready",
  countdownStartedAt: null,
  collectionStartedAt: null,
  pausedAt: null,
  pausedMs: 0,
  raf: null,
};

const el = {
  lanes: document.querySelector("#lanes"),
  phase: document.querySelector("#phaseLabel"),
  timer: document.querySelector("#timerLabel"),
  current: document.querySelector("#currentGesture"),
  trialCount: document.querySelector("#trialCount"),
  elapsed: document.querySelector("#elapsedReadout"),
  next: document.querySelector("#nextReadout"),
  subject: document.querySelector("#subjectInput"),
  block: document.querySelector("#blockInput"),
  lead: document.querySelector("#leadInput"),
  countdown: document.querySelector("#countdownInput"),
  start: document.querySelector("#startButton"),
  pause: document.querySelector("#pauseButton"),
  reset: document.querySelector("#resetButton"),
  sync: document.querySelector("#syncButton"),
  downloadEvents: document.querySelector("#downloadEventsButton"),
  downloadPrompts: document.querySelector("#downloadPromptsButton"),
  promptFile: document.querySelector("#promptFile"),
};

function nowMs() {
  return performance.now();
}

function countdownSeconds() {
  return Math.max(0, Number(el.countdown.value) || 0);
}

function leadTime() {
  return Math.max(1, Number(el.lead.value) || 3);
}

function collectionElapsedSeconds() {
  if (state.collectionStartedAt == null) {
    return 0;
  }
  const end = state.mode === "paused" ? state.pausedAt : nowMs();
  return Math.max(0, (end - state.collectionStartedAt - state.pausedMs) / 1000);
}

function logEvent(type, detail = {}) {
  state.events.push({
    event_type: type,
    browser_time_ms: Math.round(nowMs() * 1000) / 1000,
    collection_time_s: formatSeconds(collectionElapsedSeconds()),
    subject: el.subject.value,
    block: el.block.value,
    ...detail,
  });
}

function noteClass(gesture) {
  if (gesture.includes("press")) return "press";
  if (gesture.includes("release")) return "release";
  if (gesture.includes("click")) return "click";
  return "thumb";
}

function setupLanes() {
  el.lanes.innerHTML = "";
  for (let i = 0; i < laneCount; i += 1) {
    const lane = document.createElement("div");
    lane.className = "lane";
    el.lanes.appendChild(lane);
  }
}

function renderNotes(elapsed) {
  const lanes = [...document.querySelectorAll(".lane")];
  for (const lane of lanes) {
    lane.innerHTML = "";
  }
  const lead = leadTime();
  const visible = state.prompts.filter((row) => {
    const position = notePosition({ tPrompt: row.t_prompt, elapsed, leadTime: lead });
    return position >= -0.05 && position <= 2.15;
  });

  for (const row of visible) {
    const position = notePosition({ tPrompt: row.t_prompt, elapsed, leadTime: lead });
    const lane = lanes[(row.trial_id - 1) % laneCount];
    const note = document.createElement("div");
    note.className = `note ${noteClass(row.prompt_gesture)}`;
    note.textContent = row.prompt_gesture;
    note.style.left = `${118 - position * 100}%`;
    lane.appendChild(note);
  }
}

function nextPrompt(elapsed) {
  return state.prompts.find((row) => row.t_prompt >= elapsed - 0.02) ?? null;
}

function updateReadouts(elapsed) {
  el.trialCount.textContent = String(state.prompts.length);
  el.elapsed.textContent = formatSeconds(elapsed);
  const next = nextPrompt(elapsed);
  el.next.textContent = next ? `${next.trial_id}: ${next.prompt_gesture}` : "done";
  el.current.textContent = next ? next.prompt_gesture : "Complete";
}

function tick() {
  if (state.mode === "countdown") {
    const remaining = countdownSeconds() - (nowMs() - state.countdownStartedAt) / 1000;
    if (remaining <= 0) {
      state.mode = "running";
      state.collectionStartedAt = nowMs();
      state.pausedMs = 0;
      logEvent("collection_start");
    } else {
      el.phase.textContent = "PREPARE";
      el.timer.textContent = remaining.toFixed(1);
      renderNotes(-leadTime());
      updateReadouts(0);
    }
  }

  if (state.mode === "running") {
    const elapsed = collectionElapsedSeconds();
    el.phase.textContent = "RECORDING";
    el.timer.textContent = formatSeconds(elapsed);
    renderNotes(elapsed);
    updateReadouts(elapsed);
    const lastPrompt = state.prompts[state.prompts.length - 1];
    if (lastPrompt && elapsed > lastPrompt.t_prompt + leadTime()) {
      state.mode = "complete";
      logEvent("collection_complete");
    }
  }

  if (state.mode === "complete") {
    const elapsed = collectionElapsedSeconds();
    el.phase.textContent = "DONE";
    el.timer.textContent = formatSeconds(elapsed);
    renderNotes(elapsed);
    updateReadouts(elapsed);
  }

  if (state.mode !== "paused") {
    state.raf = requestAnimationFrame(tick);
  }
}

function start() {
  if (state.mode === "paused" && state.pausedAt != null) {
    state.pausedMs += nowMs() - state.pausedAt;
    state.pausedAt = null;
    state.mode = "running";
    logEvent("resume");
    tick();
    return;
  }
  state.events = [];
  state.mode = "countdown";
  state.countdownStartedAt = nowMs();
  state.collectionStartedAt = null;
  state.pausedAt = null;
  state.pausedMs = 0;
  logEvent("countdown_start", { countdown_s: countdownSeconds() });
  cancelAnimationFrame(state.raf);
  tick();
}

function pause() {
  if (state.mode !== "running") {
    return;
  }
  state.mode = "paused";
  state.pausedAt = nowMs();
  el.phase.textContent = "PAUSED";
  logEvent("pause");
  cancelAnimationFrame(state.raf);
}

function reset() {
  cancelAnimationFrame(state.raf);
  state.mode = "ready";
  state.countdownStartedAt = null;
  state.collectionStartedAt = null;
  state.pausedAt = null;
  state.pausedMs = 0;
  el.phase.textContent = "READY";
  el.timer.textContent = countdownSeconds().toFixed(1);
  renderNotes(-leadTime());
  updateReadouts(0);
}

function rowsToCsv(rows) {
  const headers = Object.keys(rows[0] ?? { event_type: "" });
  const lines = rows.map((row) =>
    headers.map((header) => String(row[header] ?? "")).join(","),
  );
  return [headers.join(","), ...lines].join("\n") + "\n";
}

function download(filename, content) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function loadDefaultPrompts() {
  try {
    const response = await fetch("./prompts_20.csv");
    if (!response.ok) throw new Error("default prompts fetch failed");
    state.prompts = parsePromptCsv(await response.text());
  } catch {
    state.prompts = defaultRows;
  }
}

el.start.addEventListener("click", start);
el.pause.addEventListener("click", pause);
el.reset.addEventListener("click", reset);
el.sync.addEventListener("click", () => logEvent("sync_marker"));
el.downloadEvents.addEventListener("click", () => {
  download("events.csv", rowsToCsv(state.events));
});
el.downloadPrompts.addEventListener("click", () => {
  download("prompts.csv", serializePromptCsv(state.prompts));
});
el.promptFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  state.prompts = parsePromptCsv(await file.text());
  logEvent("prompts_loaded", { file_name: file.name });
  reset();
});

window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    if (state.mode === "running") pause();
    else start();
  } else if (event.key.toLowerCase() === "r") {
    reset();
  } else if (event.key.toLowerCase() === "s") {
    logEvent("sync_marker", { source: "keyboard" });
  } else if (event.key.toLowerCase() === "f") {
    document.documentElement.requestFullscreen?.();
  }
});

setupLanes();
await loadDefaultPrompts();
reset();
