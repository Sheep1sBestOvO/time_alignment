import assert from "node:assert/strict";

import {
  DEFAULT_GESTURES,
  buildPromptRows,
  notePosition,
  parsePromptCsv,
  serializePromptCsv,
} from "./schedule.mjs";

const rows = buildPromptRows({
  count: 20,
  intervalSeconds: 3,
  gestures: DEFAULT_GESTURES,
});

assert.equal(rows.length, 20);
assert.deepEqual(Object.keys(rows[0]), ["trial_id", "prompt_gesture", "t_prompt"]);
assert.equal(rows[0].trial_id, 1);
assert.equal(rows[0].t_prompt, 0);
assert.equal(rows[1].t_prompt, 3);
assert.equal(rows[19].t_prompt, 57);
assert.equal(rows[0].prompt_gesture, "thumb_up");
assert.equal(rows[9].prompt_gesture, "thumb_up");

const csv = serializePromptCsv(rows);
assert.match(csv, /^trial_id,prompt_gesture,t_prompt\n/);
assert.equal(parsePromptCsv(csv).length, 20);
assert.deepEqual(parsePromptCsv(csv)[2], {
  trial_id: 3,
  prompt_gesture: "thumb_in",
  t_prompt: 6,
});

assert.equal(notePosition({ tPrompt: 9, elapsed: 6, leadTime: 3 }), 0);
assert.equal(notePosition({ tPrompt: 9, elapsed: 9, leadTime: 3 }), 1);
assert.equal(notePosition({ tPrompt: 9, elapsed: 12, leadTime: 3 }), 2);

console.log("schedule tests passed");
