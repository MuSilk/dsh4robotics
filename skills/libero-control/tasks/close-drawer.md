# Close a drawer

Push the open drawer home along its own slide axis. `SKILL.md` §7 is the mechanism (what pushes,
where to aim, how to stop, the 1 mm verdict) and §1 is the scene frame; this file is the
drawer-specific recipe, and it is written so that it holds on **either** yaw of the chest.

Everything below is in the **out-coordinate `u = S · coord[axis]`** from `rig.drawer_axis(s)`:
more `u` = further out of the cabinet, closing drives `u` down to the closed sibling's plane
`u_ref`. Two measured draws — the recipe is identical, only `S` differs:

| | KITCHEN_SCENE2 (tasks 4/5/6) | KITCHEN_SCENE4 (task 22) |
| --- | --- | --- |
| fronts' plane | y ≈ −0.229 | y ≈ **+0.216** |
| OUT | +y (`S = +1`) | −y (`S = −1`) → closing is a push in **+y** |
| open panel / home / travel | −0.0839 → −0.2447, **0.1608** | +0.0673 → +0.2183, **0.149** |
| bar band / protrusion | 0.9949–1.0047 / 0.032 | 0.9964–1.0047 / **0.0316** |
| pusher that worked | hand's face on the bar (top drawer, task 4: 7 travel + 3 stall rows) | **pitched finger tips** (bottom drawer: ~8 travel rows) |

**If `task.language` ends with "and close it"**, the drawer must stay open until the object is
inside: do the place first (`tasks/put-in-drawer-and-close.md`), then come back here.

## Recipe

1. **Probe (0 rows).** One `libero_depth`, then `rig.drawer_axis(s)` → axis, `S`, the fronts'
   plane, and the `win` window that reaches a pulled-out panel. Then, inside that window:
   - the 1 mm face histogram (`rig.face_bins`) → the fronts' bin is the biggest, the **open
     panel is the OUT-most bin with ≥200 px** that is ≥2 cm out of the fronts;
   - that panel's own z band (pixels on its exact plane, `|n| > 0.85`) → the band the verdict
     will use;
   - its **bar** (`rig.handle_bars_on`) and, above the panel, the closed drawers' bars — the
     lowest bar is the bottom drawer;
   - the closed reference plane: the biggest bin **above** the open panel's z band (a closed
     sibling's panel), *not* the frame's front face;
   - the drawer interior's colour and pixel count (`rig.colour_class_count`) while it is out.
   - Sanity: travel ≈ 0.15 m, bar protrusion ≈ 0.031, bar band ≈ 1 cm tall. If travel reads
     0.3 m or a "bar" is 20 cm wide, **the OUT sign is wrong** (§1) — stop and re-measure.

2. **Choose the pusher from geometry, not from habit.**
   - *Hand's face* if `bar_z0 − 0.094` keeps the pads above the table
     (`bar_z0 > table + 0.056`): `eef_z = bar_z0 − 0.094`, `x = bar_x_mid − JAW_DX`, the hand's
     underside on the bar band, approach OUT and push in (−u).
   - *Pitched finger tips* otherwise — and that is **every bottom drawer**: `eef_z = bar_zc −
     0.031` (measured to 1 mm on two draws), `x` such that the two open fingertips straddle the
     bar (they are 6.8 cm apart; the *pitched* jaw offset is not `JAW_DX` — read the fingers'
     own x span off one depth call), start `panel + 0.11` OUT of the bar, push in (−u). The
     hand's face cannot be used here: `eef_z = bar_z0 − 0.094 = 0.90` puts the pads under the
     table, and the wrist-down version instead stands the hand on the panel's top edge 1.3 cm
     *before* the pads reach the bar.
   - Mirror either pusher to the scene: `q_push = rig.hook_quat_toward((0, −S, 0))` (identical
     to `rig.Q_HOOK` for `S = +1`, exactly 180° for `S = −1`).

3. **Approach in the order §2 demands**: fold/rise at the *current* xy to z ≈ 1.34 → rotate
   **in place** to `q_push` and re-pass until `‖rot_err‖ ≤ 0.04` → cross in xy at that height →
   descend vertically in two stages. Measured failure of the short cut (task 22): rotation left
   at 0.08 rad, then the hover goto **stalled with a 0.15 m x-residual while `rot_err` climbed to
   0.245**, and the arm only moved when the push rows dragged it through a 30 cm sweep.
   The descent itself is **not one `goto`**: in ≤3 cm slices with a rotation-only pass between
   them, and re-read `eef_z` after each pass (a rotation-only row walks z ~1.8 cm).

3b. **Check the corridor before you commit to it — 0 rows, and it is the per-draw part.** The
   hover distance (0.11 m OUT of the bar) assumes the table in front of that drawer is empty. On
   libero_90 task 23 it was not (two leaning cutting boards, a bottle, a bowl) and the descent
   died **14 cm above the bar** with the hand — not the fingers — on a board's top edge: all axes
   dead, `+z` alive, nothing within 4 cm of the eef in the depth, `rot_err` climbing. `rig.above_table_map(s)` lists the clutter; `s.arm_yz()` shows the hand's own ridge (at a pitched
   pose it is **0.19 m behind and 0.015 m below** `eef_pos`, so the blocker is *behind* the eef).
   If the column is dirty, use the three legs that worked there: **descend at a y where the palm
   is still in front of the clutter → cross OUT (toward the camera) at a z that is still above the
   drawer front's top edge → descend the last few cm** (there the palm passes *under* the leaning
   boards). 17 rows, and the push then went home first try.

4. **Buy the push for 0 rows.** At the pre-push pose, one depth call: the out-most arm pixels in
   a z window around the bar are the fingertips. Their z should straddle `bar_zc` (±8 mm) and
   their x span should cover the bar. Correct `eef_z`/`eef_x` once from that reading, then push —
   that is what makes the tips constant transfer between draws.

5. **Push and stop on the panel's own plane.** A saturated row along `−u` with x/z servoed and
   the quaternion held every row; every ~4 rows one snapshot and
   `rig.front_face_on(s, fr, band_z)`; break when `−6 mm ≤ u_panel − u_ref ≤ +1.5 mm`. A stall
   (3 rows < 1.5 mm) is the back-up, not the rule: on task 22 the last 8 rows moved the panel
   0.149 m at 8–19 mm/row and never stalled. Then one `+z` row to prove the dead rows were the
   drawer and not a freeze.

6. **Verdict, from the final depth call with the arm clear** — all four, and they agree:
   1. the panel's own plane by its **z span** (`rig.panel_home_on`), asymmetric: home is at
      `u_ref` or 1–2 mm **deeper**; never 2 mm out;
   2. the **interior's pixels collapse** (`rig.colour_class_count`: 6 169 → 0 on task 22, so it
      also works when the interior is the arm's colour and `planes()` cannot see the floor);
   3. the **bar back on the closed plane**: `u_bar == u_ref + dz_bar` within ~2 mm (task 22:
      0.1860 measured against 0.1847 expected);
   4. **nothing in front of the fronts' plane** in the histogram except the bars' own sliver
      (open state's panel bin at +0.067 gone, i.e. no face at `u > u_ref + 5 mm`).
   A plane test alone can be fooled by the hand occluding the panel; 2 and 4 cannot.

7. **Retreat along OUT first, then up.** The push ends with the fingertips 3 cm in front of the
   frame *on the bar*, but a bad approach can leave them past the fronts' plane. The fingers
   point inward, so a straight move `u − 0.24` at the same z slides them out below the bar;
   rising first would drag it.

## Failure modes, in order of frequency

| symptom | cause | fix |
| --- | --- | --- |
| stalls 2–5 mm proud, weak `+z` rise | the hand's face was above the panel band, riding the frame / counter edge | drop 4–5 cm and push the **bar**; on a low drawer switch to the tips |
| the arm travels 0.15 m and the bar's plane never moves | pads/fingers were below the bar, pressing the closed drawer underneath, or in the panel-bar pocket | re-measure the panel plane *first* (a stalled eef is no evidence), then re-aim by the **measured bar band** |
| every plane reads plausibly but is 0.15–0.30 m off; a 22 cm-wide "bar"; travel 0.31 m | the chest's OUT axis is not +y | `rig.drawer_axis` → use the `_on` helpers; the **sign is the bug**, never add an offset |
| the goto stalls with a big residual on one axis while `rot_err` > 0.2 | the rotation was left half done | re-rotate to ≤0.04 rad, then rise → cross high → descend vertically |
| **the descent stops 10–15 cm above the bar, `+z` still moves, nothing is near the fingers** | the **hand block** is resting on table clutter *behind* the eef (a leaning board, a bowl, a bottle) — not a freeze, not the drawer | do not sweep the height (5 mm moves a 0.19 m lever by nothing): re-measure with `s.arm_yz()` / `rig.above_table_map`, then change the corridor (step 3b) |
| panel home but the hand is inside the cabinet | a sweep carried the tips past the fronts' plane | leave along OUT first, then rise |
| home reads 2 mm *past* the reference | not a fault | a shut drawer homes 1.5–2 mm deeper than its sibling on both measured chests |

A clean close is **~20 travel rows + 2–4 stall rows + 1 `+z` row + 1 depth call**; the whole
task-22 episode (probe, approach, push, retreat, 4-signal verdict) was **155 rows**.
`code/close22/controller.py` is that run, and `ENTRY=selftest CONTROLLER=code/close22 python3
code/lib/run_controller.py` re-derives every number above **offline** from the saved npz files —
0 rows, no episode — so the metrology can be re-checked before you spend a single row.
