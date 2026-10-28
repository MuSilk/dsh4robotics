# Open a drawer

Task 6 (`KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet`) is this subtask: pull a
drawer out of the counter-height chest that stands on the table. The scene is **redrawn every
session**, so this file is a procedure and not a table of numbers: each *placement* value is
obtained by a 0-row depth call plus the derivation beside it, and every claim that the drawer
moved is a plane measurement. The *asset* values (z bands, travel, the wrist pose that bites)
do transfer and are listed in §2/§3 — command those, do not re-derive them.

**Scene frame first (0 rows, `SKILL.md` §1).** `rig.drawer_axis(s)` gives the bank's axis, the
**OUT sign** `S` (drawers open toward the camera side of the fronts' plane) and the fronts'
plane. KITCHEN_SCENE2's chest faces `S = +1` (+y out); KITCHEN_SCENE4's faces `S = −1` (−y out,
so every `+y`/`−y` in this file mirrors). Write the relations in `u = S · coord[axis]` and use
`front_face_on` / `handle_bars_on` / `panel_home_on` — the `+y`-only helpers return *plausible*
numbers 0.15–0.30 m out rather than failing.

Budgets: the hook is **~45 rows of motion** once you know the numbers (~41 pre-hook + rotate +
offset read, ~29 press, ~16 pull), and the whole open-with-verify is ~120 rows. The measured
pass with **no demo file** was **632/1000** including ~225 rows of avoidable diagnostics; §3 is
what that episode should have done. A close-and-reopen cycle (§7) is +~80 rows and is only owed
if you replayed a demo trace — with the HDF5 gone, you cannot, so you do not.

## 0. The silent failures, newest first

1. **A z-held pull is not a probe of a bar.** Measured here, seconds apart on the same pose:
   `rig.probe_pull` (saturated +y with z servoed, 4 rows) returned **move = 0.000 m** at
   eef_z 0.950, 0.955, 0.965 and 0.971 — *including* the pose from which an 8-row burst of
   `dy +0.85 / dz −0.35` pulled the panel **4.4 cm**. The −z bias is what makes the fingers hold
   a handle bar. Use `rig.probe_pull_biased`, and read its **`dz_actual`**: on the bar the
   commanded −0.35 is blocked (0.000–0.002 m); off the bar the full −0.029 m is taken and the
   tips walk down the panel's face.
2. **At the hook pitch `pads()` reads the PALM, not the fingers.** At eef_z 1.0993 (15 cm above
   the table), boxes 0.030/0.040/0.045 gave **+0.0606 / +0.0586 / +0.0586** — flat, so it passes
   every guard in `rig.pitch_offset` — and it is 3.5 cm above the real finger surface. A press
   derived from it sits the fingers over the bar. The true number came from the arm (§2): the
   finger's top is `eef + 0.0197`.
3. **The press height has a 5 mm window, and it fails upward.** At `eef_z` 0.955–0.975 the press
   converges in y with a ≤1 mm residual (nothing touched) and the pull moves the panel 0.000 m;
   at 0.9499 it stalls with a 10–12 mm y residual and the first burst moves 4.4 cm. Do not
   sweep ±1 cm — that cost one episode ~180 rows and never touched the bar (§3).
4. **A pull tells you nothing while it happens.** Measured on an earlier attempt: a pull whose
   fingers sat 1.6 cm above the bar ran **0.198 m** of +y at ~12 mm/row, no dead row, the
   quaternion servo converged, the picture showed a plausible pull — and the drawer never moved.
   The pull that worked travelled 0.162 m with the *same* absence of stall. ⇒ never buy the
   travel rows before a probe has shown the drawer's own front plane moving.
5. **"Front-most" is not "biggest".** `rig.y_faces_mm` returns bins **sorted by count**, so
   `rows[0]` is the largest face in the band — on an opened drawer that is the chest's own
   back/side face. Use `rig.front_face_y` (max y) and keep `min_pixels ≈ 150`: at 40 a 67-pixel
   noise bin sitting on the closed plane won the front-most test on a drawer already 0.159 m out.
   The fronts are seen **edge-on** here (the camera's optical axis has no y component), so one
   drawer's panel really is only ~600 pixels — do not raise `min_pixels` to "clean it up".

## 1. Measure the chest — 0 rows, one depth call

```python
s = rig.snapshot(ctx, tag="probe0")                 # 0 rows
xybox, band_z, counter_z, counter_y = rig.chest_window(s)
closed_front = rig.front_face_y(s, band_z=band_z, xybox=xybox, min_pixels=150)[0]
bars    = s.handle_bars(panel_y=closed_front,
                        zrange=(s.table_z() + 0.005, counter_z - 0.01), xbox=xybox[0])
bottom  = min(bars, key=lambda b: b["zc"])          # the BOTTOM drawer = the LOWEST bar
drawer_band = rig.drawer_band_for_bar(bottom)       # that drawer's own z band
own_ref = rig.front_face_y(s, band_z=drawer_band, xybox=xybox, min_pixels=150)[0]
bar_front = bottom["front_y"]                       # closed: use it
DZ_BAR  = bar_front - own_ref                       # ~+0.0322: the bar stands this far proud
```

| step | why it works | check |
| --- | --- | --- |
| `chest_window` | the counter's front edge sits within ~2 mm of the fronts' plane, so it anchors **both** the front band (above the table objects) and the y window (which must reach `counter_y + 0.22` or a pulled-out panel falls outside it) | window covers the chest's x span; band is empty of table objects |
| `front_face_y` | the chest's drawers are **one flush front plane** — that plane, not any drawer, is the closed reference | it is a big bin (hundreds of px); with `min_pixels=150` no bar leaks into it |
| `handle_bars` | the bars are the only grey things standing **proud** of that plane, one per drawer, ~8 cm apart, so the **lowest bar is the bottom drawer**. This is the drawer's identity — the fronts cannot tell the drawers apart | one bar per drawer (3 on this chest); each bar's front is ~3 cm proud of `closed_front`; bars ordered in z |
| `drawer_band_for_bar` | widens the bar's z band 4.5 cm down / 0.6 cm up: that drawer's panel, and not the next drawer's | band contains `bottom["z0"]..bottom["z1"]` and stops below the next bar |
| `own_ref` | this drawer's own closed front plane, in its own band | within ~2 mm of `closed_front` |
| `DZ_BAR` | **the bars move only in y**, so once the drawer is out `bar_front = panel_front + DZ_BAR`; this is the only way to find the bar on a moved drawer | `DZ_BAR` is 0.030–0.034; re-check it on every new draw |

**Never call `handle_bars(panel_y=<the open drawer's plane>)`.** The bars are achromatic
(mean 75–115), i.e. inside `Scene.arm`, and the search window then opens *in front of* the open
panel — where the open **gripper** sits. On an earlier attempt it returned a "bar" with front
`+0.0759`, z 0.996…1.038 and x −0.14…+0.057, and the close pre-pose went 0.28 m away from the
drawer and pushed air for 5 rows. Use `panel_front + DZ_BAR`, and keep the bar's x span and z
band from the closed probe (they do not move).

One draw, for **scale only** (everything here is redrawn; this one measured the same values as
the draw this file was last verified on, which is a coincidence, not a promise): table 0.9498;
fronts −0.2432; bar front −0.2110, z bands 0.9949–1.0047 / 1.0644–1.0735 / 1.1394–1.1469,
x −0.0728…+0.0162 (mid −0.0283); bottom drawer's own band (0.9499, 1.0107); counter 1.175 with
its front edge −0.2421.

## 2. A wrist-down grasp of a bar is impossible — pitch the wrist, and measure two numbers

The hand block reaches **0.111 m ahead** of `eef_pos` with its underside at `eef_z + 0.082`
(measured again here: `hand_lead` = 0.1114, `finger_lead` = 0.0632). A wrist-down pose low
enough to put the pad band on a low bar buries the hand's face inside the drawer front, and
every descent stops with **all axes dead while `droll` still turns** — SKILL §2's third freeze
test, not a controller freeze.

**The hook orientation is a measured constant — use it, the trace is gone:**

```python
# the orientation the working demos settled into (SKILL §10.1); 15 deg away from the
# idealised (1,-3,-2)/(1,0,0) build, and that 15 deg is what decides contact:
#   the idealised press slid past the bar twice (0.4 mm residual, panel moved 0.000 m)
q_hook = rig.Q_HOOK          # == rig.quat_from_fingers((0.260,-0.855,-0.455), (0.933,0.10,0.34))
# CHECK the builder itself once: finger (0,0,-1) + jaw (0,-1,0) must reproduce the
# MEASURED wrist-down quaternion to <=5 deg (3.6 deg here).  Worse means the column
# convention is wrong.
```

**Two numbers to measure at the pitch, and neither comes from `pads()`:**

* **the finger's top**, with the arm itself (§0.2): press into the handle recess, then command
  pure `+z` rows with x servoed and read the stall. Measured **1.04 cm of rise, stall at
  eef_z 0.9752** against the bar's underside at 0.9949 ⇒ **`finger_top = eef + 0.0197`**, about
  half the wrist-down `PAD_DZ`. ~5 rows. `rig.finger_top_offset` does this.
* **the press height**, from the bar not from the fingers: `eef_z = bar_z0 − 0.045` = **0.9499**
  on this asset (bar band 0.9949–1.0047), which leaves the finger's top 2.5 cm *below* the bar's
  underside — a corner jam, not a face contact, and lower than anything a pad-offset derivation
  would give.

**Do not press at the ascent's stall height.** 0.9752 (finger top exactly on the bar's underside)
was itself measured to converge and pull nothing; the contact that works is a corner jam 1.5 cm
lower, with the finger's top 2.5 cm *below* the bar. The ascent tells you where the fingers are;
the press comes from the bar. Both facts are needed and neither replaces the other.

The palm and forearm sit 10–15 cm **up and back** (+z, +y) from the fingers, which is why a
press at 0.955–0.975 clears the bar entirely.

**Rotate in place first** (SKILL §2): rotation-only rows, `action[3:6] = rotvec_err/0.30`. The
pitch is ~100° from wrist-down, driven at a 0.08–0.10 rad tolerance in **~18 rows**, and the eef
drifts ~5–10 cm while it turns, so goto afterwards.

## 3. The recipe — the numbers that were measured to work

```python
x, bf = m["bar_x_mid"], m["bar_front0"]
rig.goto(ctx, [x, bf + 0.16, 1.00], rig.Q_HOOK, -1, max_rows=14)      # 10 rows, clear
rotate_in_place(ctx, rig.Q_HOOK)                                      # 18 rows, in place
z = m["bottom_z0"] - 0.045            # 0.9499: the ONLY height that bites (5 mm window)
payload, conv = press(ctx, x, bf - 0.015, z, max_rows=26, stall_rows=5)   # ~21 rows
```

1. **Press DIAGONALLY onto the bar's top-front**, `−y` and `−z` together, to
   `(bar_x_mid, bar_front − 0.015, bar_z0 − 0.045)`. The pocket between the bar and the panel is only
   ~1.3 cm deep, so aim *past* the bar's front face: `bar_front − 0.005` reached y −0.2138…−0.2159
   and converged without biting, while `−0.015` (−0.2275 here) stalled at y −0.2169. Measured
   stall poses on the draw that passed: **(−0.0272, −0.2166, 0.9516)** and
   (−0.0261, −0.2169, 0.9507).
2. **Require the y residual: 10–12 mm is contact, 0–1 mm is a miss** — and the residual alone
   is **not** the verdict: gate on the **realised stall**. Print `stall_y − bar_front` (in `u`)
   and require **≥5 mm**. Measured on libero_90 task 23 with the same 1.5 cm aim: a stall
   **1.2 mm past** the bar's front face passed the residual test (13.8 mm) and the biased probe
   then moved the panel **0.0006 m** with `dz_actual −0.023`; re-approaching so the stall landed
   **10.7 mm past** it gave **4.0 cm** on the first 6-row probe with `dz_actual 0.0000`. So press
   deeper rather than higher, and print the stall's depth.

2b. **Score the probe in `u`.** `probe_pull_biased` returns `move_u = S·Δy` next to the raw
   `move`: on a −y-opening chest the raw value is **negative for a drawer that opened** (task 23:
   `move = −0.0402` = 4.0 cm of travel, reported as "not engaged" by a raw test). `dz_actual ≈ 0`
   is necessary, not sufficient — a tall bar can block the tips without catching them.
3. **Probe with the pull you will run** — `rig.probe_pull_biased`: 6 rows of `dy +0.85 / dz
   −0.35`, jaws open (−1), x servoed, panel plane before/after. Pass = panel moved ≥ 5 mm
   **and** `dz_actual ≈ 0` (measured 4.07 cm and −0.0011). Fail = the full −0.029 m of z taken
   and move 0.000: re-press **5 mm lower** (0.945, then 0.940), never 1 cm higher.
4. **Pull to the STOP, not to "open enough"** — 8-row bursts of the same profile, the drawer's
   own plane checked after each burst. Measured: 4.4 cm → 6.3 cm → 5.9 cm = **0.1609 m out**
   (two bursts after the probe's 4.07 cm). The slide joint has **0.1605 m** of travel and the
   predicate is a fraction of that, so do not stop at 0.11 m. The eef stays at eef_z ~0.950
   through the whole pull because the bar blocks the commanded −z — if z starts falling, the
   fingers have left the bar: re-measure the panel, set `bar_front = panel + DZ_BAR` and re-press
   (`open_once` loops exactly this).
5. **Verify**: retreat **+z 0.10 then +y 0.13** (the arm occludes the drawer), then one depth
   call and three signals (0 rows):
   * the drawer's own band's **front-most** 1 mm y-face moved from `own_ref` — require
     **≥ 0.14 m** (measured 0.1594);
   * `handle_bars(panel_y=own_ref, reach=0.30)` — a bar now sits in front of `own_ref + 0.10`
     with the same z band you hooked (that is *which* drawer moved);
   * the closed-front window (`reach=0.09`) now holds one bar fewer than `len(bars)` did.

   Two of the three plus the panel signal is a pass. **The bar *count* is a weak signal**: it
   drops the bar nearest the gripper (a 0.12 m eef-distance filter inside `handle_bars`), so a
   correctly closed drawer reads 2 bars instead of 3, and a wrongly-found "bar" can read 3.
   Trust the panel plane; use the bars for "which drawer". The picture settles the rest: the
   box is visibly out with the drawers above shut.

## 4. Failure tells

| symptom | cause | fix |
| --- | --- | --- |
| press converges with a ≤1 mm y residual, pull move 0.000 **and z falls the full −0.03** | press too high: the fingers cleared the bar (§0.3) | drop to `bar_z0 − 0.045`; if it was already there, 5 mm lower |
| press stalls in y (10–12 mm residual) but the biased probe moves nothing | aimed too shallow in y: the fingers stopped in front of the bar's back face — **check `stall_y − bar_front`: ≤2 mm is the tell** | press to `bar_front − 0.015`, not −0.005, and print the realised stall depth |
| the descent dies 10–15 cm above the bar, `+z` alive, nothing near the fingers | the **hand block** is on table clutter behind the eef (0.19 m behind, 0.015 m below at the hook pitch) | audit the corridor (`rig.above_table_map`, `s.arm_yz`) and come in where the palm is clear → cross at the working z → descend |
| z-held probe moves nothing but the eef travels its full length | the probe does not match the pull (§0.1) | use `probe_pull_biased`; the bias is the mechanism |
| biased probe move 1–4 mm | partly engaged, or the drawer was already at its stop | check `own_ref` first: it may already be open |
| pull's z falls to ~0.93 and the panel does not move | no contact to hold the tips: they are walking down the panel's face | re-press (§3.1), keep the bias only while the panel moves |
| `pads()` at the pitch is flat but ~0.058 | the palm, not the fingers (§0.2) | use the ascent test / the bar, never this for the press height |
| `pads()` negative at the pitch | the TABLE is in the box (eef_z ≲ 0.99) | measure with the eef ≥ 12 cm above the table |
| a "bar" appears at `y +0.07`, z outside a bar band, x span > 0.15 | `handle_bars(panel_y=<open panel>)` caught the gripper | `bar_front = panel + DZ_BAR` (§1) |
| front-most bin is < ~150 px, or not a plane you recognise | window/band wrong: table object inside the window, or the y window stops short of the travel | re-derive with `rig.chest_window`; print `y_faces_mm` and read the histogram |
| all axes dead on a descent near the chest, `droll` still turns | hand's face (`eef_z + 0.082`) on a bar top or the frame | pitch the wrist (§2); do not kick |
| the *middle* bar moved | you aimed at the wrong bar | the bottom drawer's bar is the lowest one (§1) |

## 5. Budget

| run | rows | notes |
| --- | --- | --- |
| the hook alone, numbers known | **~45** | 0 probe; 10 pre-hook + 18 rotate + ~7 offset read; 21 press; 6 probe; 16 pull; 16 verify |
| task 6, pass, **no demo file** | **632/1000** | 41 hook, **88 on three presses at 0.955/0.965/0.971 that converged and touched nothing**, 137 diagnostics (differential depth, the ascent test, three `try_pull` probes), 29 press that worked, 16 pull, 21 verify, 28 close, ~50 reopen, 16 final verify |
| task 6, pass, with a replay | 931/1000 | 137 replay, ~300 blind height sweeps, 24 close, 45 reopen, 43 re-pull |

Of the 632, ~190 were avoidable with §§0–3 as they now stand: the working press height, the
biased probe and the deep press y were all derivable before a row was spent. Spend the rows on
the pull instead — it is 16 of them.

## 6. If a demo file ships

`tasks/<task>_demo.hdf5` is being removed and nothing here depends on it any more: the numbers
it teaches live in **SKILL §10.1** (travel 0.1605 m, `rig.Q_HOOK`, press at `bar_z0 − 0.045`,
the `dy +0.85 / dz −0.35` profile, `DZ_BAR` 0.0322, the 1.5 mm home plane). If the file *is*
present, mine it (SKILL §10.2) for the joint range, the orientation and the action profile, and
if you replay any of its rows, do the close-and-reopen cycle (§7) before `finish`.

## 7. The close-and-reopen cycle (only if you replayed a trace)

Replaying is learning; the final state has to be your own motion. With the HDF5 gone there is
nothing to replay, so this cycle is not owed — `ENTRY=run` in `code/open_drawer2` therefore stops
after the open + verify, and `ENTRY=run CYCLE=1` runs the cycle when you want it. Both halves are
measured to the millimetre:

```
A  close with the finger tips           28 rows   panel -> -0.2447 (1.5 mm PAST own_ref)
B  reopen: measured press + biased probe ~50 rows   panel -0.2447 -> -0.0823 (0.1609 m out)
```

### A. Close it with the finger tips — 28 rows

The SKILL §7 recipe (push with the hand's face, `eef_z ≤ bar_top − 0.088`) **cannot be used on
the bottom drawer**: its bar band is 0.9949–1.0047, only 5 cm above the table, so the hand's
face would need `eef_z = 0.917` — 3 cm *under* the table. Push with the finger tips at the hook
pitch you already hold:

```python
z = bottom["zc"] - 0.031                      # 0.9688: tips (rel +0.030) land in the bar band
rig.goto(ctx, [x, panel_open + 0.10, z + 0.09], rig.Q_HOOK, -1, max_rows=24)   # approach clear
rig.goto(ctx, [x, panel_open + 0.10, z],        rig.Q_HOOK, -1, max_rows=16)   # descend
# then saturated -y rows, x and z servoed, panel checked every 6 rows
#   stop when |panel - own_ref| <= 2 mm, or the eef stalls 3 rows
```

Measured on the draw that passed: panel −0.0838 → **−0.2447** (1.5 mm past `own_ref` −0.2432, in
24 push rows), the push stalling at eef y −0.216. The verdict is the panel plane alone; the
closed-window bar count read 2 vs 3 on a drawer that was physically shut.

**The first bursts are free travel.** The eef starts ~0.10–0.13 m in front of the bar, so the
panel reads *unchanged* for the first ~10 rows — the same reading you would get from a press
aimed at nothing. Do not stop on "the panel is not moving" until the eef has stalled at least
once.

### B. Reopen it — ~50 rows

1. rotate in place to `rig.Q_HOOK` (0–4 rows: after the close the wrist is already there);
2. `goto (bar_x_mid, bar_front + 0.16, 1.00)`, 10 rows;
3. **diagonal press** to `(bar_x_mid, bar_front − 0.015, bar_z0 − 0.045)`, 21–26 rows. Derive the
   y from the *measured* panel: it comes home **1.5 mm past `own_ref`**, so
   `bar_front = panel + DZ_BAR` = −0.2125 and the target is −0.2275, which stalls at
   **(−0.0261, −0.2169, 0.9507)**;
4. **biased probe**, 6 rows: panel −0.2447 → −0.204 (4.07 cm), `dz_actual` −0.0011;
5. **two pull bursts** of 8 rows: −0.204 → −0.141 → **−0.0823** (0.1609 m out).

Skip the shallow `bar_front − 0.005` press here: on the reopen it landed at y −0.2138 and the
z-held probe read 0.000, i.e. the same razor it is on the first hook, and it cost 55 rows to
find out. One number decides it — the y residual of the press — and the biased probe confirms.

### C. Verify, then finish

`+z 0.10`, `+y 0.13` (tolerance 0.014, 8 rows each way), one depth call, then: the panel at
`own_ref + ≥0.14` (measured 0.1594), the lowest bar's front ≈ `panel + DZ_BAR`, and the picture
— the drawer box visibly out with the drawers above shut.
