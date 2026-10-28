# Turn on the stove — and the pair clause that comes with it

"Turn on the stove and put the frying pan on it" (libero_90 task 21, `KITCHEN_SCENE3`) is **two
clauses** and they fail independently:

```
(:goal (and (turnon flat_stove_1) (on chefmate_8_frypan_1 flat_stove_1_cook_region)))
```

Measured on one episode (938 rows): `turnon` **passed** with the knob at +37°, and the pan clause
failed with the pan's base plane *exactly* on the raised burner (0.9803 vs 0.9804), tilt 0.4°,
rim fit `R 0.0985` / arc 358° / res 1.4 mm — because its **centre was 7.5 cm off the cook
region** (`(0.0263, 0.1493)` against the burner's `(−0.052, 0.180)`). The finish verdict names
the region: **`on(..., flat_stove_1_cook_region)` is not "resting on the plate"**. So: turn the
knob here, and read `tasks/put-on-surface.md` for the pan, with its new "region" and "carry"
sections.

## 1. What the knob is (the object task 18's notes call "a black decoy pan")

It is the stove's **knob**: a vertical-axis joint with a raised fin. Measured on this draw:

| what | value | how (0 rows) |
| --- | --- | --- |
| hub: flat disc, r90 ≈ 0.043, **top z 0.9720** | a horizontal face ⇒ **joint axis vertical** | mode of z over the **dark** blob below the fin (the plate is 2 mm higher — match colour, not just z) |
| fin on top of it: **7.8 × 2.6 cm, top face z 1.0089**, level | the handle you turn | tight z band + `\|n_z\| > 0.85` |
| axis | fin centroid ≈ (−0.2228, 0.1837) (the hub's own centroid is biased ~1 cm by the fin's occlusion) | |
| fin azimuth at rest | **178.9° ⇒ −1.1°** (an *axis*, mod 180) | `rig.flat_face_az` |
| stove plate / raised spiral | plate **0.9740** (x −0.156…0.032, y 0.090…0.280), raised spiral **0.9804** at **(−0.052, 0.180)** | `rig.plane_mode` + the pixels > plate + 2.5 mm |
| **the fin's THICKNESS vs `s`** (the one that decides whether a push does anything) | a **0.4 mm sheet** for `\|s\| < 0.030` (its −n face at **`n = −0.0119`**, flat from s = 0.005 to 0.050), and a **24 mm-thick block** over **`s = +0.030…+0.050`** (n −0.012…+0.012, r ≈ 0.044) that **rotates with the fin** | `n = p2` of the fin-frame `n` coordinate, ±8 mm window per 5 mm of `s`, at the pad band (`z = hub_top + 0.008 … fin_top − 2 mm`) |

## 2. The state sensor: an angle, and a lamp (both 0 rows)

* `rig.flat_face_az(scene, knob_axis, fin_top)` → the fin's azimuth. The fin is symmetric about
  the axis, so this is an **axis angle (mod 180)**: use `rig.az_delta(new, old)` for the change,
  and never wrap a *continuous* angle you are feeding into a quaternion (wrapping flips which
  side of the fin you are on and the whole contact geometry with it).
* **The asset lights its burner rings red when it is on.** `rig.red_pixels(path, box=(280, 390,
  320, 475))` reads **0 px off → ~1440 px on**, and it lit by ~22° of knob rotation. That is a
  free, unambiguous `turnon` verdict before `finish` — use it. (Once the pan covers the burner it
  reads 0 while the stove is still on; test it *before* the pan lands.) Counting the **whole
  frame** is enough — there is nothing else red in the scene — and the burner's own pixel box on
  the third draw was v 288–342, u 342–421, so the box above is a safe superset.
* The on-threshold is **low**: lit at ≤22°, and the clause passed at +37°. The joint is **not**
  spring-returned — the fin stayed at +36° for the rest of the episode (and at +51° on the draw
  that finished).
* **`flat_face_az` degrades as the fin turns, so do not re-measure a late azimuth as a verdict.**
  The top face goes edge-on to the camera and is partly occluded by the pads: n = **258 px at
  az +8°**, **142 px at +51°**, and **111 px on the final re-read, which returned +71° for a fin
  that was physically at +51°** (the PCA of a partial, edge-on face). Read the azimuth *between
  pushes* (arm lifted) and gate the pass on the **lamp** you saw on the way up, not on a fresh
  angle after the pan has landed.

## 3. What does NOT turn it (two silent failures, ~120 rows between them)

| approach | measured result | why |
| --- | --- | --- |
| **pinch the fin and yaw the tool** (jaw-pair centre on the fin's centreline) | a **+25°** yaw moved the fin **+2.2°** | the jaw-pair centre is (−0.019, −0.013) from `eef_pos`, so the tool's yaw axis sits **23 mm off the knob's axis**; and two flat faces can only carry a rotatable bar by friction |
| **rotate the whole tool rigidly about the knob's axis** with a finger face pressing the fin's face (10° steps, closed loop on the fin) | fin moved **0.0°** per step | a face-on-face contact transmits force along its **normal, which is radial at the contact ⇒ zero torque** about the axis; τ = −F·s only for a contact offset *along* the fin, and the position servo relaxes the 1 mm penetration in one row |
| squeeze-then-open the jaws | fin jumped **+22°** (the pads' closing sweep pushed it) | this is a **corner** contact — the same mechanism that works, just uncontrolled |

Every one of these "converges" (the rotation error goes to ~0.03) and does nothing. **The eef's
own rotation is not evidence that the knob turned**; only the fin's azimuth is.

## 4. What turns it: a corner push, ratcheted

The working contact is a **fingertip corner/edge on the fin's side face, offset along the fin**
(so the normal force has a lever arm), applied with the jaws **open** and one finger:

* pose: pad band **0.990…1.012**, i.e. `eef_z = hub_top + 0.010 − PAD_DZ ≈ 0.94` (10 mm above the
  hub's top face, 5 mm below the fin's top); the finger's face beside the fin's side face at a
  **`|s|` where the fin is THICK** — see §8: `|s| = 0.040` on the draw that finished, because the
  blade is a 0.4 mm sheet inside `|s| = 0.030` and only becomes a 24 mm block beyond it.
* direction: contact on the **−u end** pushing in −n ⇒ **CCW** (τ = −F·s, s<0); the **+u end** ⇒ CW.
  On this asset **CCW is "on"** (fin −1.1° → +36.2°). Equivalently: **the +u end pushed in +n is
  also CCW**, and that is the side the camera can see (the −y face), so prefer it: contact at
  `axis + s·u`, pad's inner face starting ~6 mm outside the fin's measured −n face, push along
  `+n = (−sin az, cos az)`.
* what worked in one shot: two **16 mm straight pushes** (from +y moving −y, then from the other
  side) — 37° total, and the rings lit. Uncontrolled but cheap.
* the controlled version — **ratchet**, closed-loop on the measured azimuth:
  1. measure the fin (`rig.flat_face_az`), pick the contact side for the direction you want;
  2. `goto` the push pose (hover, then descend; the pads pass beside the rim, so aim the jaw
     centre on the fin's centreline and the *finger* at the contact point). **Yaw the tool to the
     fin's azimuth** so the pads close along the fin's `n` — then the pad's face is parallel to
     the fin's face whatever the angle;
  3. push **in metres, servo'd** (30 mm of commanded travel; its residual is the contact sensor:
     4–8 mm left is contact, 0–1 mm is empty air);
  4. measure: `az_delta` ≥ ~5° = it turned; **< 3° twice in a row = a stop or a miss**;
  5. **back off ~20 mm along the push axis**, then lift 3 cm and re-position along the *new*
     azimuth, descend, repeat.
  2–4 cycles for 40–60° once the contact is on the fin's thick end. Budget **~70 rows**
  (approach 30, two pushes + one depth call each, retreat 10).

Do not locate the joint's stop by pushing: with a face contact "no motion" is geometry, not a
limit. +37° is already enough for `turnon`; stop there and spend the rows on the pan.

## 5. Do not disturb the knob afterwards

The pan's rim must clear it. At the placement target the pan's rim's −x edge is
`target_x − R = −0.151`, and the fin's +x end is at ≈ −0.19: **~4 cm of clearance**, and the fin
is the tallest thing there (top 1.009, above the pan's base plane at 0.98). Aim the pan at the
**cook region** and the clearance takes care of itself; aim it "somewhere on the plate" and the
pan lands 7 cm +x, which both misses the region and eats that margin.

## 6. Failure modes

| symptom | cause | fix |
| --- | --- | --- |
| yaw "converges" (rot_err → 0.03) and the fin does not move | friction/slip pinch, or a radial face push | §3: a corner push with an offset along the fin |
| `fin: no flat top face (0 px)` | the z band was taken from max(z) over the neighbourhood — a dark pad pixel 6 mm higher sets it | a **fixed** band around the probe's `fin_top`, `\|n_z\| > 0.85`, radius < 0.07 |
| the fin's azimuth is stuck at ±90° | the axis angle is mod 180 and you wrapped it | keep it continuous; use `az_delta` for changes |
| the burner never lights though the fin moved | you turned it CW (the off direction) from the start | the first 10–20° decide it: no lamp + no fin motion ⇒ reverse |
| the pan lands on the plate but `on(...)` is False | the clause names `*_cook_region` and the pan's **centre** is off the burner | `tasks/put-on-surface.md` §"the clause may name a REGION" |

## 7. Measured the hard way: one episode, 1000/1000 rows, BOTH clauses false

A later run of this same task spent **~860 rows on the knob and never lit the lamp**
(+22.4° best, ending at −2°), then ran out of rows with the pan still on the table. Every
number below is measured on a draw whose geometry matched this file's to ~1 mm (axis
(−0.2227, 0.1841), fin top 1.0089, hub top 0.9720 r 0.0432, plate 0.9740 x[−0.156, 0.031]
y[0.091, 0.279], raised burner 0.9799 at (−0.0597, 0.1847) r 0.060). Read it before the
first knob row; it is worth more than the whole of §4's budget.

**The fin's usable face is much shorter than its top plate.** The top face is ~6.6 cm long
(PCA `ext` 0.038), but at the pad band (`eef_z + 0.042 … + 0.064`, i.e. z 0.982…1.004 with
`push_z` 0.9403) the blade only exists out to **|s| ≈ 0.022–0.025 m from the axis**:

| contact | result |
| --- | --- |
| `\|s\| = 0.0228` (0.60 × ext), 12–15 mm of −y travel | fin turned **+2–3° per push** (9 cycles → +22.4°) |
| `\|s\| = 0.0285` (0.75 × ext), **33 mm** of −y travel | fin turned **0.0°** — the pad swept 12 mm *past* the fin's measured +y face and touched nothing |

So: keep `|s| = 0.022`, do **not** scale it with `ext`, and if a push that travels well past
the measured face moves nothing, the pad is missing the blade — move the contact *toward*
the axis, do not push harder.

**Command the push travel in metres, and make it ≥ 20 mm.** At the knob pose
(x ≈ −0.23, y ≈ +0.25, z ≈ 0.94) the eef's y stride collapses to **1.4–6 mm/row**
(10–13 rows for 15 mm), against 12–13 mm/row for a free goto elsewhere. `rig.push(...,
rows=3)` therefore moved 12 mm and only just reached the fin — that is the whole reason the
first 215 rows produced +1.3°. Servo on `y`, not on a row count, and print the residual: a
residual of 4–8 mm after the loop *is* contact.

**Never flip the direction on a weak result.** `E*P = +1` is CCW (contact at `x < axis_x`
with a −y push, or at `x > axis_x` with +y). A "3 cycles with no motion ⇒ reverse" rule
fired while the fin was in fact turning 1.4–2.9°/cycle and cost **25° of accumulated
rotation** (the fin went +23 → −2). Reversing is only justified if *the lamp stays off and
the fin does not move at all* over 10–20° of commanded travel.

**Back off along the push axis before lifting.** Lifting 7.5 cm while the pad was still
against the fin's face dragged the fin **7° backwards** (measured: az +18.9° at the end of
one stage, +11.9° at the start of the next, with nothing but a vertical retreat in
between). `-push * 0.022` in y, then up.

**The camera cannot see the fin's +y face at all.** `camera_to_base` puts the camera at
x = +0.637 looking toward −x/−z, so the fin (long axis along x) shows its **−y side face**
as a 1.5 cm sheet and its +y side is hidden. Depth at z 0.985…1.000 returns dark pixels in
a *single* 1 cm y bin (0.17–0.18) across x −0.26…−0.19 — that is the −y face, not the fin's
outline. Take the **+y face from the top-band pixels' `max Y` in the pad's x window** (it
reads 0.193–0.195 and the fin is 1.9 cm thick in y), and treat any `min Y` of the full
blob as the visible side, i.e. 2 cm away from the face you must push.

**Two 0-row segmentation traps, both of which cost a probe run:**
* `~scene.arm` excludes the **stove plate** (pale grey is achromatic with `mean > 70`).
  Find the plate as the *mode of z* over light-grey pixels in (table+0.015, table+0.040),
  x > −0.45 (the far cabinetry is at x ≈ −1.4 and also reads table-height z).
* The **burner is dark and 5 mm above the plate**, so it wins "the tallest dark blob near
  the plate" over the knob. Search the knob only *behind* the plate: x < plate_x0 + 0.01.

**And the pan, for the record (this draw):** centre (−0.0653, −0.2771), R 0.0970, rim
plane 0.9863, `h_pan` 0.0365, tilt 0.8°, `R_base` 0.0675, handle rising to **z 1.084** at
the +y side (so the −y lip is the only pinch, and the carry must clear 1.08). `robust_ring`
**fails** on the standing pan — its handle is the same colour and rises above the rim, so
the top band is the handle (measured R 0.152, res 10 mm). Use a coarse (cx, cy, R) Hough on
a z band of `table + h_pan ± 9 mm`, then two trimmed LSQ passes ⇒ R 0.0970, res 1.8 mm,
tilt 0.8°, 360° of arc. At the burner centre the pan's rim comes within **2.8 cm** of the
knob's fin (+x end at −0.185), so the region target and the knob clearance agree.

**Budget, in one line:** with §4's recipe working, the knob is ~60–90 rows and the pan
grasp + 0.46 m carry + region descent is ~230; check `libero_status` after every stage and
abort anything that has spent 2× its estimate — 860 rows on a knob is an episode.

## 8. The draw that finished it: +53° in two pushes, 776 rows, both clauses true

Third draw of this same scene, geometry equal to the two above to ~1 mm (axis (−0.2227, 0.1841),
fin top 1.0086, hub top 0.9720, plate 0.9740 x[−0.156, 0.039] y[0.070, 0.279], raised burner
0.9800 at (−0.0597, 0.1847) r 0.060). What it cost, and the two rules that turn it into ~70 rows:

**The same fin, two contacts, a 5× difference.**

| contact | travel commanded | result |
| --- | --- | --- |
| `\|s\| = 0.022`, the 0.4 mm sheet (residual 15–23 mm = "contact") | 30 mm | **+5.4, +2.9, +1.0, +0.7°** — 163 rows for +10°; the pad slips past a knife edge and each push catches less |
| `\|s\| = 0.040`, the 24 mm block at the fin's +u end | 30 mm | **+13.9°, +31.1°** — two pushes, ~50 rows, lamp 0 → **1444 px** |

So `|s|` is not a fraction of the top plate's extent and it is not the number in §7 — it is
**where the blade is thick at pad height**, and that is measured in the probe with an `n`-face /
thickness profile (§1's last row). Both parts of the fin rotate together (the block's polar
angle tracked the fin's azimuth: −2.9° → −0.2° → +1.7° as the fin went −1.9° → +3.5° → +8.1°),
so the thick end is a legitimate handle.

**Never let a probe re-run at a low eef, and never let it redefine the rest azimuth.** One probe
was re-run with the arm parked beside the pan at `eef_z` 0.99: `pads()` returned the **stove
plate** (−0.0199 instead of +0.0417) and `jaw()` a jaw centre 9 cm away (+0.073, −0.007 instead of
−0.020, −0.015). Every knob pose afterwards was **7 cm out in x** and three pushes travelled their
full 32 mm through empty air (residual −0.001 m, "EMPTY AIR"). Two free guards: gate the probe on
`eef_z > 1.10`, and keep the **recorded** `az0` when the fresh reading is more than a few degrees
away — the probe reads the fin's *current* angle, so a re-run after 10° of progress silently
redefines "turned 0°" (this happened, and made the log report "turned −0.8°" on a knob that was
at +9.2°).

**The push pose, in one line** (`rig.pad_dz` measured per draw; here +0.0417):

```
jaw  = axis + s·u + (n_face − 0.006 + gq)·n        s = 0.040, n_face = −0.0119, gq = 0.034
eef  = jaw − Rz(az)·(jaw_dx, jaw_dy)               jaw_dx, jaw_dy = −0.0204, −0.0153
q    = yaw_quat(q_down, az)
eef_z = hub_top + 0.010 − pad_dz                   = 0.9403  (pad band 0.982…1.004)
push  = +n for 30 mm                               n = (−sin az, cos az)  (CCW)
```
then `−n·0.022`, `+z 0.045`, one depth call: fin azimuth + `red_pixels`. Two cycles.
