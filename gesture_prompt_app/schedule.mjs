export const DEFAULT_GESTURES = [
  "thumb_up",
  "thumb_down",
  "thumb_in",
  "thumb_out",
  "thumb_click",
  "index_press",
  "index_release",
  "middle_press",
  "middle_release",
];

export function buildPromptRows({
  count = 20,
  intervalSeconds = 3,
  gestures = DEFAULT_GESTURES,
} = {}) {
  return Array.from({ length: count }, (_, index) => ({
    trial_id: index + 1,
    prompt_gesture: gestures[index % gestures.length],
    t_prompt: index * intervalSeconds,
  }));
}

export function serializePromptCsv(rows) {
  const body = rows.map((row) =>
    [row.trial_id, row.prompt_gesture, formatSeconds(row.t_prompt)].join(","),
  );
  return ["trial_id,prompt_gesture,t_prompt", ...body].join("\n") + "\n";
}

export function parsePromptCsv(csvText) {
  const lines = csvText
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return [];
  }
  const headers = lines[0].split(",").map((header) => header.trim());
  const required = ["trial_id", "prompt_gesture", "t_prompt"];
  for (const field of required) {
    if (!headers.includes(field)) {
      throw new Error(`prompts.csv missing required column: ${field}`);
    }
  }
  return lines.slice(1).map((line) => {
    const values = line.split(",").map((value) => value.trim());
    const row = Object.fromEntries(headers.map((header, index) => [header, values[index]]));
    return {
      trial_id: Number(row.trial_id),
      prompt_gesture: row.prompt_gesture,
      t_prompt: Number(row.t_prompt),
    };
  });
}

export function notePosition({ tPrompt, elapsed, leadTime }) {
  return (elapsed - (tPrompt - leadTime)) / leadTime;
}

export function formatSeconds(value) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}
