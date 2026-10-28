# Put an object onto a raised surface ("put the frying pan on the stove")

The payload is **bigger than the jaws' 6.8 cm span**, so the only place it can be pinched is its
**lip**, and a lip-pinched payload with its COM far from the pinch **hangs tilted**. That tilt is
the whole task: at 28° a 20 cm pan's base droops 3.5 cm below its own base plane, which turns a
careful placement into landing it on one edge and walking it off a 19 cm target. Measured on
libero_90 task 18: **202 rows, PASS**, end state pan centre (−0.056, +0.185) on the stove, base
0.9808 on the 0.9800 burner, tilt 0.4°, and the release did not move it.

**Task 21 is this placement plus `(turnon flat_stove_1)`**, and its goal reads
`on(chefmate_8_frypan_1, flat_stove_1_cook_region)` — a **named region**, not the plate. One
episode finished with the pan's base **0.9803 on the raised burner 0.9804, tilt 0.4°** and still
got `False`, because the pan's **centre was 7.5 cm off the burner** (see "the clause may name a
REGION"). The `turnon` half is a knob on the stove's back: `tasks/turn-on-stove.md`.

Read this after `SKILL.md` §2 (the levelling paragraph), §4's "surface the object must rest on",
"lowest point" and "carried black object" rows, and §4c.

## What the scene is (task 18, one draw — placement numbers never transfer)

| what | how to get it | this draw |
| --- | --- | --- |
| the pan's rim circle + plane | dark mask, **4 mm** top band, circle+plane fit — **but the handle rises ~10 cm above the rim, so its top band IS the handle** (task 21: `robust_ring` gave R 0.152, res 10 mm). Fit the rim with a coarse (cx, cy, R) Hough on a band of `table + h_pan ± 9 mm`, then trimmed LSQ (`pan_fit`) | c (−0.0776, −0.2492), **R 0.0981**, rim plane 0.9870, tilt 0.4° |
| the pan's height `h_pan` | `lip face − base plane` while it stands — **the lip's face, not the dark-band fit's `zc`** (that reads 3–4 mm inside the bright outer edge and 4.6 mm low: 0.0348 against 0.0393 on task 21) | **0.0372** (table 0.9498); **0.0393** on task 21's draw |
| the pan's **base outer radius** `R_base` | the r(z) profile of the silhouette | **0.075** at z 0.955 → 0.098 at the lip (a 38° cone — the lip is the only pinchable part) |
| the pan's wall thickness at the lip | radius percentiles of the top band | annulus r **0.0957–0.1007**, i.e. 5 mm — but this is the *asset's* number: task 21's draw has a **wider, bright rolled lip** (rgb 97–103, top face out to r ≈ 0.1005, its bright part starting at r ≈ 0.083), and that brightness is why a dark mask tops out 2 cm below it |
| the **support surface** | `zmax` over the target's own region | plate 0.9747, but the **raised spiral burner 0.9800** (r ≈ 0.076) is what the pan rests on |
| the target's top face | bright/grey plane at the support z | x −0.160…+0.032, y 0.090…0.280 (19 × 19 cm) for a 20 cm pan |
| what is in the way | other blobs near the target | the moka pot (x 0.02…0.07, top 1.06) sits in the carry band; a **7 cm black decoy pan** at x < −0.165, top 1.009 sits behind the target |

**A pan has no foot ring**, so unlike the bowl there is nothing to bracket with two descents —
but its *base* radius does matter (§4c, the tilt correction) and the r(z) profile gives it in
one call. The rim numbers repeat to ~0.5 mm between fits of the *same* standing pan (an
independent fit of this one read zc 0.9874, tilt 0.6°, R 0.0985, i.e. `h_pan` 0.0376): take one
fit, print it, and command that number for the whole episode — mixing two fits is a 0.4 mm
error for free. `objects_on_table` returns **nothing** in this scene: a black pan and a grey
stove are both outside its chromatic, mid-bright class, so segment by hand
(`mean<85 & chroma<30` for the pan, `mean>90` at the support z for the stove).

## Grasp: the −y lip, wrist down, no rotation

The jaws close along **y**, so a circular pan's rim can only be straddled at its ±y extremes.
The **+y** extreme is where the long handle leaves the pan (the handle rises to 1.03 there, and
its root blocks the pads); the **−x/+x** arcs cannot be pinched at all. Take the **−y** lip with
`rig.rim_grasp_target(..., side=-1.0, dz=-0.055)`; the pan then hangs on the eef's **+y** side,
which is also the side the carry has to go.

* Approach: fold up to `z 1.30` **first**, then cross to the lip, then hover 10 cm above it.
  Task 18's target was eef (−0.057, −0.330, 0.932) at y −0.33, where `j4` was −1.64 and the arm
  tracked to 2 mm — but only after the fold-up (§2). 58 rows for the whole approach.
* Pad band: `eef_z = rim_plane − 0.055` puts the pad bottom ~12 mm below the lip, which on this
  flared wall is the **top 1.2 cm of the cone**. Descend with the jaws open and the pads
  straddling the wall — nothing else is on the way, because the hand's underside stays 8 cm
  above the rim at all times.
* A close that settles at **`gq 0.0098` is not a miss on a cone** (§3): that is a 19.6 mm gap on
  a 5 mm lip, and the pan was held for the whole episode. Judge the grasp from the ring fit one
  snapshot later, never from `gq`.

## Measure the payload the moment it is off the table

Lift 0.13 and fit the **carried** pan with the §4c window. This is 0 rows and it is the number
every later stage needs: `d_xy` (centre − eef), `d_z` (zc − eef) and `tilt`. On task 18's first
lift it read tilt **27.5–28.3°**, `d_xy` (−0.026, +0.075), `d_z` +0.013, R 0.0984 — and the
pan's own decision was already visible in it: 3.5 cm of droop on the low edge.

## A tilted pan's fit is a slice of the cone — `zc − h_pan` is then 2 cm low

**This is the one that fails silently, because the fit is beautiful.** On a *level* carried 20 cm
pan (task 21, third draw, after the levelling rotation) `carried_black_ring` + `robust_ring`
returned `R 0.0831, zc 1.0849, res 0.7 mm, arc 222°, tilt 0.1°` — reproducible to 0.2 mm and
**not the rim**. It is the 38° flared wall at exactly the height where it emerges from behind the
pads; the pan's lip is at r ≈ 0.099 and z 1.112. Releasing on `zc − h_pan` would have buried the
pan ~2 cm in the stove, which reads afterwards as "flat on the burner, still False".

Why the usual guards do not catch it:

* **A z-band sweep cannot separate them** — a horizontal cut of a level cone is also a circle, and
  the wall slice is often the *cleaner* circle (0.7 mm against the lip annulus's 1.8 mm), so
  "best residual" and "most ring pixels" both pick the slice. Prefer the **highest** qualifying
  ring inside a narrow window around a *prediction*, and gate `R > 0.095` (the lip is the widest
  feature: `R_outer` ≈ 0.1005, the slices are 0.083–0.095).
* **A dark mask cannot see the lip at all**: the pan's lip is **bright** (rgb 97–103) while the
  interior and the shadowed outer wall are rgb ≈ 44. So "the top of the dark blob" is 1.5–2 cm
  below the lip, and `robust_ring`'s top band is a wall slice by construction.

What *is* unambiguous, both measured on the **standing** pan in the same 0-row probe and then used
unchanged on the carried one:

| feature | how | standing pan (this draw) | carried pan |
| --- | --- | --- | --- |
| **lip face** | `p99` of z over the annulus `r 0.085…0.112` about the fit's centre, **handle side cut** (`y < cy + 0.085`) and **pads cut** (a 45 mm cylinder about the *measured* jaw centre — the pads are bright grey and sit in that annulus) | **0.9893** ⇒ **`h_pan = lip − base = 0.0393`** | 1.1141 ⇒ base **1.0748** |
| **interior floor** | `p10` of z of the dark pixels at `r < 0.055` (the mouth faces up, the floor is always visible) | **0.9519** ⇒ **`floor_dz = 0.0019`** | 1.0731 ⇒ base **1.0712** |

They agree to 3.5 mm, so take their mean and print both. Note the dark-band Hough's
`zc − table` = 0.0348 is **not** `h_pan`: it reads the lip 3–4 mm inside its bright outer edge and
4.6 mm low. `carried_base` / `lip_fit` in `code/stove21/controller.py` are this section.

Two free bonuses from the same numbers: `base − eef` stays **+0.013** in the grip and
`centre − eef` stays **(−0.044, +0.083)** through a lift, a 29° levelling, a 0.5 m carry and a jam
— so both the final aim and the final height can be dead-reckoned; and `tilt = 2·acos|q·q_hold|`
is the pan's tilt (the pinch is a clamp), so the low edge is `base − R_base·sin(tilt)` without
touching the depth.

## Level it, then thread `q_hold`

`rig.level_quat(n, q_cur)` gives the tool rotation that takes the measured normal to +z; drive
it with `rig.rotate_to` (gated on `rot_err`), ~11 rows for 28°:

```
28.3° → 3.7° in 11 rows, eef walked only 3–4 mm
```

then **persist the achieved quat** (task 18 wrote it to `experiments/pan18/plan.json` and every
later stage loaded it). A later `goto` with `Q_DOWN` re-tilts the pan 28° in a single move — that
is the same trap §2 records for the bowl's 17° lock. After the rotation the pan **tracks the
wrist**: 3.5° through the whole 0.5 m carry.

## Carry and place

* **Carry high enough for the tallest obstacle in the band.** The pan's base sits `d_z + ... ` —
  on task 18, at eef_z 1.13 the pan's base was 1.16, clear of the moka pot's 1.06. Aim the *pan's
  centre* at the target (`eef_xy = target − d_xy`) and re-measure: the pan creeps in the grip
  (`d_z` 0.0608 → 0.0527 over one carry), so one alignment pass after the carry, at 4 mm
  tolerance, is not optional.
* **A long carry drifts in x, and the drift is not correctable at the end.** Measured on task 21:
  over a 0.45 m +y carry at eef_z 1.135 the eef slid **+8 cm in x**, the orientation lost
  **0.27 rad** of `q_hold`, and the `goto` ended **STALLED** with `j4 = −0.070`. So re-derive
  `xy = target − d_xy` from the carried pan **at every descent step**, not once at the start.
* **When `dx = −1` is inert, print `j4` — and do not assume it will be at the "stretched" end.**
  Task 21 jammed at (x ≈ 0.03, y ≈ 0.10, z ≈ 1.25) with `j4` pinned on its **−0.07** limit, i.e. at
  a *short* reach near the base, while the pan's own far reaches read −1.19 and −2.16. 20 rows
  moved a 4–8 cm x error by only 0.5 mm/row. The cure is `SKILL.md` §2's anyway: **rise** —
  z 1.30 was not enough, **z 1.42** was, and the same −x target then ran **5.6 cm at
  6–16 mm/row**. Rise, reset the quaternion, re-aim from the measured pose.
* **Never release on loop exhaustion.** The stride collapses to 2–6 mm/row near the folded/edge
  poses: a 46 cm descent covered 30 cm in 34 rows, and a closed loop capped at 14 × 1 cm then
  opened the jaws **17 cm above the stove** — the pan fell onto the plate's **+x edge** and stayed
  there. Gate the release on the *measured* `base_low ≈ support + 1 mm` at that pose; if the loop
  is running out of rows, **rise and re-aim, never open**.
* **The order that works from a jammed carry is level → rise → cross → descend, one question per
  stage** (task 21, third draw, 776 rows, PASS). Each step exists because the previous one is
  what blocks the next:
  1. **level in place** — `rotate_to(q_hold)` gated on `rot_err`; a **0.30 rad orientation error
     stalls every axis** (the OSC trades orientation for position), and the pan is clamped, so
     levelling the tool levels it. This was the difference between "all axes inert at 1.31" and a
     clean descent.
  2. **rise at the CURRENT xy** to `z 1.42` (`j4` went −0.07 → −1.66 and the elbow folded);
  3. **cross in xy at 1.42** — 14 rows for 7 cm of `−x`, which is impossible at 1.30;
  4. **descend vertically in two stages** (coarse to `z_t + 0.03`, then to `z_t`) — never a
     diagonal that descends and crosses in one `goto`, which re-stretches the elbow and stops
     with the xy error still 12 mm.
* **The release height can be dead-reckoned, and that saved the episode.** With
  `base − eef = +0.013` and `tilt = 2·acos|q·q_hold|`:
  `eef_z = support + 1 mm − 0.013 + R_base·sin(tilt)` → 0.9772 here → the pan landed with base
  **0.9801** on the 0.9800 burner, tilt **0.7°**, centre 17 mm off the cook region. At a 4.8°
  carrier tilt that correction is 6 mm; skipping it is a 6 mm bury.
* **You cannot push a mis-placed pan back onto the burner.** With the pads at 0.99–1.01 the arm
  cannot reach far enough +x (needed `eef_x` 0.146 at z 0.94; it stalled at 0.139, `j4` −0.07) to
  put a fingertip outboard of the pan's +x rim, so the slide push never engages — twice, the pan
  moved 0 mm. There is no cheap recovery: the aim has to be right the first time.
* **Aim the target to clear the KNOB** (task 18's notes call it "a 7 cm black decoy pan"; it is
  the stove's own knob — a vertical-axis joint with a 7.8 × 2.6 cm fin whose top is at z 1.009,
  `tasks/turn-on-stove.md`): the pan's rim reaches `target_x − R`; task 18's target x −0.055 put
  the rim at −0.154, 1.1 cm clear of the knob at −0.165, and its base (r 0.075) still inside the
  stove's 19 cm face. Task 21's fin's +x end sat at −0.19, 4 cm clear of the rim at the burner
  centre — the clearance is adequate **only** at the burner, not 7 cm +x of it.
* **Descend on the LOWEST point**: `goal = support_zmax + 1 mm`, and each step
  `z_t = eef_z + (goal − base_low)` with `base_low = base − R_base·sin(tilt)`
  (`rig.base_low`). One coarse goto on the predicted height, then 3–5 closed-loop steps of
  ≤8 mm with one depth call each — 40 rows all told.
* **Release in place, then retreat `+z`.** Open the jaws over 14 rows holding the pose, then
  +0.14 m. Opening did **not** shove the 20 cm pan off (unlike the bowl's 3.8 mm, §3): the pan
  settled from base_low 0.9801 to 0.9808 and its tilt dropped 2.6° → **0.4°** as it seated.

## The clause may name a REGION, not the plate

`finish`'s false-clause list is the only spec you get, and on task 21 it read
`on(chefmate_8_frypan_1, flat_stove_1_cook_region)`. The referent is a **cook region** — in
practice the burner — and the test is on the **pan's centre**, not on "is it touching the stove":

| state | value | verdict |
| --- | --- | --- |
| pan base, on the raised burner | 0.9803 vs 0.9804 | — |
| pan tilt | 0.4° | — |
| rim fit | R 0.0985, arc 358°, res 1.4 mm | trustworthy |
| **pan centre** | **(0.0263, 0.1493)** vs the burner's **(−0.052, 0.180)** | **False** |
| pan centre inside the *plate's* bbox (x −0.156…0.032) | yes, by 6 mm | irrelevant |

So **aim the pan's centre at the measured burner centre and require
`|centre − burner_centre| < 0.02 m` on both axes** before `finish`. No amount of flatness,
contact or "it is on the plate" rescues a 7 cm offset, and the pan's base (r ≈ 0.073) already
overhangs an 18.8 cm plate by ~1 cm at a perfect centre — the tolerance is small and there is
nothing to spend on the rim.

## Verify, then finish

From the final depth call, with the arm clear: the pan's **centre within 2 cm of the burner
centre** (above), its base plane on the support surface (±2 mm), tilt small, and the source spot
empty (`max z` over the old footprint back at `table_z`). Task 18 read base 0.9808 / plate 0.9747
/ spiral 0.9800 / tilt 0.4° / centre over the face true. `on(pan, stove)` is a support/**region**
clause, so **release** is the right ending — unlike the stack, which has to stay held (§0.7).

For the *released* pan use `pan_state` (in `code/stove21/controller.py`): a Hough on an absolute
band `support + 0.026 … + 0.050` (the lip is at `base + h_pan`), same trimmed-LSQ pass, then
`base = p99(annulus) − h_pan`. Task 21's third draw read, after the release and a 6-row `+z`
retreat: centre (−0.0464, 0.1739) against the burner (−0.0597, 0.1847) ⇒ off (13.3, 10.7) mm,
base 0.9801 against 0.9800, tilt 0.7°, source spot 0 px above `table + 2 cm`, and `finish`
returned `task_complete: true`. `red_pixels` reads ~0 once the pan covers the burner — that is the
pan occluding the lamp, not the stove going off, so the `turnon` verdict must be the one you took
before the descent (here 1444 px at +20.3° of knob rotation).

## Failure modes, in order of frequency

| symptom | cause | fix |
| --- | --- | --- |
| the payload lands on one edge, tips, slides off | placed on the base plane while carrying a 20–30° tilt | level it (§2) *and* descend on `base_low` |
| `pan_ring` / `robust_ring` returns `None` on a held pan | the blob is pan + its own shadow + the wrist's dark blocks | the §4c z window; widen `Rlo/Rhi` to the payload |
| the carried fit is *clean* (`R 0.083, res 0.7 mm`) and the pan still buries / hovers 2 cm | the ring is a **slice of the 38° cone**, not the lip (§"A tilted pan's fit") | base from the **lip annulus** (`p99`, handle+pads cut) and the **floor** (`p10`), averaged |
| every axis inert with `‖rot_err‖ ≈ 0.3` and `j4 ≈ −0.07` | the OSC trades orientation for position; the elbow is straight | **level in place** (`rotate_to`), *then* rise at the current xy, *then* cross, *then* descend |
| the payload re-tilts between the fix and the place | a `goto` with `Q_DOWN` after the levelling rotation | pass the captured `q_hold` to **every** row |
| the payload hovers a few mm over the target | descent aimed at the plate rather than the **raised** support | `zmax` of the target's region |
| flat on the stove, contact right, `on(...)` still False | the clause names a **region**: the pan's centre is off the burner | aim the **measured burner centre**, ±2 cm |
| the pan ends 5–8 cm +x and the arm cannot move it | the long carry drifted x, and with `j4` pinned on −0.07 (a short reach near the base) `dx = −1` is inert | re-aim every descent step; **rise to z 1.42** before any −x |
| the pan lands on the plate's edge, tilted | released because the descent loop ran out of rows | release only on measured `base_low ≈ support + 1 mm` |
| an off-centre pan and a slide push does nothing | the arm cannot reach outboard of the pan's +x rim at pad height | there is no recovery — aim right the first time |
| the payload catches the object standing behind the target | target aim chosen without the pan's radius | `target_x ± R` must clear it (that object is the stove's KNOB — never push it) |
| the deliverable is the wrong black disc | two pans in the scene | "the frying pan" is the one with the long flat handle + hanging hole; the 7 cm disc is not |
