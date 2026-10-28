---
name: libero-explore
description: Write one LIBERO controller from depth-referenced 3-D targets. imajev_decide names objects; depth measures them. Trial it with run, reset onto a random scene, and submit. The method below was worked out on the butter-to-tray task.
---

# LIBERO manipulation skill

A round is already open. Read `TASK.txt` for the instruction, the 2400-step exploration budget, and this round's cap. Write `code/<name>/controller.py` defining `main(ctx)`. Trial that file with `run`, read `score`, then `reset` and `run` again. `reset` takes no arguments and opens a random scene; nothing tells you which one. When the same program works on another scene without edits, `submit`. After that, it is rolled out on every initial scene of every task in this session. `TASK.txt` says whether that is one task or all of LIBERO-90. There is no `finish`.

Read the instruction from `libero_status` at the start of every `main`. When the session covers every task, `reset` changes the instruction as well as the scene. Do not hard-code the first one. The butter numbers below are one example of the method.

`run` spends environment steps, up to this round's cap (`remaining_episode_steps`, at most 400) and up to the 2400 shared across rounds. `reset`, `score`, cameras, and depth do not spend steps. Steps after `submit` do not spend the 2400. The controller cannot call `score`, `reset`, `run`, or `submit`, and it is not told the budget.

Working code is already in `code/` — read it before writing anything:

| file | what it gives you |
|---|---|
| `code/mcp.py` | unix-socket JSON-RPC: `connect`, `tool(ctx, "libero_step", ...)` |
| `code/geom.py` | `components(mask)` (no scipy here), `quat_to_mat`, `rotvec` |
| `code/rig.py` | depth frame, `solve_delta`, `scene_objects`, `goto`, `descend`, `close/open`, `unwind` |
| `code/pickplace.py` | `pick_at`, `place_into`, `lift_to` |
| `code/inspect.py` | pose, DELTA + checks, table height, one 4× crop per object |
| `code/vlm.py` | imajev client: `available()`, `decide(questions, images)`, `answer(body, key)` |
| `code/ask.py` | one imajev call from the shell: `python3 code/ask.py crop_a.png crop_b.png` |
| `code/scene_id.py` | `identify` (which object), `best_container` (tray), `container_floor` |
| `code/scene.py` | deprecated shim — its old `T_CAM` was wrong; never unproject with a hardcoded camera |
| `code/template/controller.py` | the `main(ctx)` shape to copy |

Run `python3 code/inspect.py` from the shell while you are looking, before `run`. It opens its own connection. Inside `main(ctx)`, pass that same `ctx` into `rig.solve_delta`, `rig.goto`, and `rig.tool`. Do not call `rig.connect` from the controller.

**imajev is in this session.** The tool is `imajev_decide`, and `code/vlm.py` talks to the same socket. It names an object in one or two pictures; it does not return metres. Use it when geometry leaves more than one candidate, then measure the chosen cluster with depth. Read `abstained` and `unknown_probability` (`vlm.answer` returns `(None, p)` on an abstention) and fall back to the geometric sort. The first call of a session loads the model, about two or three minutes, so make it before a grasp. A controller reaches the same socket through `IMAJEV_SOCKET`; pass the `ctx` you were given into `rig`, and call `vlm.decide` from ordinary Python.

## The one thing that will bite you: DELTA

Depth comes back in a camera frame that is *not* the base frame the arm reports. The
rotation is good — an unprojected table patch is flat to ~3 mm across the whole image — but
the translation is offset by a constant per episode:

    eef_pt = depth_pt + DELTA          # solve once, first thing, at a home-like pose

`rig.solve_delta(ctx)` does it: take the light-grey arm blob that hangs in from the top of
the picture (`rgb.mean>105 & chroma<28`, `z∈[0.25,1.05]`, blob starts at row 0, narrower
than 250 px), average the bottom-3-row pixels (the two finger tips), `DELTA = eef_pos - tip`.

Trust it only after both checks print green:
1. **finger separation ≈ 0.078 m** — the two tips are the jaws, which are 0.078 m apart open.
   This single number validates the whole unprojection (it was 0.083 m on this rig).
2. **table lands at 0.38–0.48 m** after adding `DELTA_z` (0.429 on one scene).

Pitfalls that cost two wasted calibrations: the *table* is the largest near-depth blob
(it is only 0.65–0.70 m from the camera, same as the gripper), and the background wall is
also light grey. Use the row-0 / width / depth-band filters above.

On one scene DELTA was **(-0.517, 0.015, 0.426)**. It stays put for that round and changes
after `reset`. The controller must call `solve_delta` at the start of every `main`. Do not
hardcode DELTA, and do not copy object coordinates from a previous round. Older notes
(`-0.60, 0, +0.42`, butter at `(-0.01, 0.06)`, tray `x∈[-0.22,0]`) are other scenes of the
same task.

## Manipulator

- 7-D OSC deltas in the base frame. Full scale ≈ **0.010 m/step**; the first step of a burst
  is ~4× smaller. P control `row[:3] = clip((target-p)/0.06, ±0.9)`, 5 steps per block,
  re-read the pose every block: ~4 cm per block, lands within 5 mm, never jammed a joint in
  219 steps.
- `goto` early-exits at `tol=0.008`, i.e. it can stop 5 mm short — use `tol=0.004` for a
  release height.
- Jaws close along the eef **y**. At home `R[:,1] ≈ (0,-1,0)`, already right for these
  boxes — no spin needed. `gripper_qpos` 0.034 = open (0.078 gap), **0.0197 = holding
  something**, 0.005 = air closed, 0.023 = object still brushing the pads after a release.
- Rotations only if the jaws are crosswise; then `unwind` (`[0,0,0,0,0,+1,-1]` repeated) both
  frees a jammed wrist and spins the jaws.
- `libero_observe` and `libero_step` return a **flat** dict (`eef_pos`, `eef_quat_xyzw`,
  `gripper_qpos`, `joint_pos`, `step_id`, `remaining_episode_steps`) — no `observation`
  wrapper. `remaining_episode_steps` is this round's cap. The shared 2400 is `status` →
  `budget.used` / `budget.total`; a controller does not receive those fields.
- A stall is the top of whatever is under the pads. **A 7×4 cm box straddled by an open jaw
  does not stall the descent** — the pads pass either side and you stall on the table. "No
  stall at the object height" is not a miss; the jaw reading after closing is the real check,
  and the wrist view at the grasp pose shows the object right between the fingers.

## Camera geometry

- agentview and wrist are 512×512, `fx=fy=618.04`, `cx=cy=256`, depth `uint16` mm, row 0 top.
  Depth pixels are aligned with the agentview RGB (verified to ~1 px), so crop and unproject
  the same pixels.
- The agentview sits ~0.65–0.70 m from the gripper and looks along **−x, ~32° below
  horizontal**: x motion barely moves the image, it mostly changes depth. You cannot eyeball
  x alignment — use the depth, then confirm under the wrist camera.
- The arm leaves the top of the agentview once the eef is above ~0.80 m or behind ~−0.20 m
  in x (it projected to v=−24 at `(−0.21, 0.05, 0.86)`). Solve DELTA with the gripper fully
  in frame.
- The wrist camera sees the fingers along the bottom edge and the grasp area just above
  them; at grasp height ≈ 38 px/cm (a 3.7 cm butter ≈ 140 px wide). Best place to read a
  label: crop it from the agentview and upscale 4×, or read it in the wrist view when the
  hand is over it.

## Task recipe (butter → tray, one scene)

Read the instruction from `libero_status`. The numbers below are measurements from one
scene of that task. Objects move on every `reset`. Use the procedure; recompute the numbers
inside `main`.

1. `libero_status` → instruction. `solve_delta` → DELTA, table_z.
2. `scene_objects` → clusters (`z > table+0.012`, wall/horizon blobs filtered). Crop each,
   upscale, read the label. On that scene the orange box was **Breakstone's "FRESH BUTTER"**
   (target, centre `(0.063, 0.051)`, top 0.462, 7.0×3.7 cm) and the blue/white stick was
   **Philadelphia cream cheese** (decoy); also a ketchup bottle, two cans, and the wooden
   tray. Find them again from the crops. Do not reuse the centre.
3. Hover `top+0.14`, descend with the jaws open to **`top−0.012`**, close 8 steps → jaw
   about 0.0197, lift 0.13, look at the wrist view.
4. Container target: `rig.floor_region(pts, z, table_z, delta, bbox=tray_bbox)` gives the
   interior floor point, the floor height and the wall ranges. On that scene the floor was
   0.453 and the walls were `x∈[−0.155,0.068]`, `y∈[0.225,0.368]`. Carry at **z ≥ 0.58**
   (rim was 0.48–0.52), lower to about **0.02 above the floor**, open 6 steps, retreat
   straight up 0.14.
5. Check the geometry yourself: the object's cluster xy is inside the container's floor
   `xr`/`yr` *and* its top ≈ floor + object thickness (on that scene the butter's label top
   read 0.469 after landing on the 0.453 floor). Then call `score`. A picture of the object
   sitting in the tray is the second check; a single yes/no vision answer is not enough.
   The controller cannot call `score`, so it has to make this geometric check itself before
   it returns.
6. `reset`, then `run` the same file. Submit once it succeeds on a second scene without you
   editing the coordinates.

The motion itself is about 85 steps once DELTA is right. Two bad calibrations are what
burned a 219-step round. If a round passes ~150 steps and the grasp has not happened,
`reset` and fix the controller instead of spending the rest of the 2400 by hand.

## If the grasp misses

Jaw < 0.010 after closing means air. Open, drop another 1 cm, close once more (that retry is
built into `pick_at`). If it misses twice, the xy is wrong — re-read the object's cluster
(its xy moves only if you knocked it) and compare with the wrist view before trying again.
Never translate with a saturated command for dozens of steps; check `joint_pos` if a push
stalls (a pinned `q4`/`q6` means unwind, healthy joints mean reach).
