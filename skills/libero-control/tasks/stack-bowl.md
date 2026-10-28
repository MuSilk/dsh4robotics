# Stack one object on another ("stack the black bowl at the front on the black bowl in the middle")

Three identical speckled bowls, one plate, no drawer to open: a grasp, a carry, one placement
and a verdict. It is the shortest task in this suite, the only one where the **final state** is
the whole verdict — and the only one where the **identity of the two objects** can be lost with
no depth signal at all.

`SKILL.md` has the method. This file is the distilled version of six episodes of tasks 16/17
(four failed on the state, one on the pair, one passed).

## SOLVED — task 17 PASSED, 272/1000 rows, ending HELD (episode #5)

`finish` returned `task_complete: true` with the jaws still closed on the bowl:

| upper base plane | lower rim | margin | tilt | lateral | source spot | contact |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00001 | 1.00076 | **−0.75 mm** | 1.1° | 40 mm | empty | **yes** — Δbase 0.67 mm short of Δeef, destination took the load |

**A held pose at the rim plane IS `on`.** Do not spend a row on a set-down or a release: every
released state measured on these bowls is worse (see "what does not work" below). The recipe:

1. **`probe` (0 rows).** Find the three bowls, `rig.rim_plane` each, `rig.bowl_height` for this
   draw, and the **destination's annulus** from the radius histogram of the top 1.5 mm of its
   rim. x orders them: `front = max x, middle = median x, back = min x`.
2. **`grasp` the middle bowl's `-y` rim**: `rig.rim_grasp_target(rim, side=-1.0, dz=-0.055)`,
   then `close_grip` until the jaws settle. One pinch, `gq` 0.0036–0.0043, ~46 rows for the whole
   over / pre-down / down / close / lift. Verify the source spot's `zmax` collapsed to the table.
3. **Level it immediately** — before any lateral move, while the tilt is still small:
   `rig.rotate_to(ctx, q_hold, eef)` gated on `rot_err`, then **capture the achieved quaternion
   and pass it as `q_tgt` to every later row**. Measured here: 14° → 1.7° in 4 rows, and it stays
   level only while that quaternion is held (a single `goto` with `Q_DOWN` re-tilts it 17°).
4. **Aim `d = 0.030` in +y**, i.e. `aim_xy = dest_centre + d·ŷ`, and put the **measured** carried
   centre there (`eef_xy = aim_xy − (centre − eef)`), with the goto tolerance at **1.5 mm** — 4 mm
   is enough to lose the aim. `d` is the whole placement: the foot ring has to reach the annulus,
   so `d > annulus_inner − r_foot` (0.0255 here), and the draw that worked used **~4 mm more than
   that minimum** (0.030). At 0.027 the base follows the arm to the rim plane with **no contact**
   (the foot is entirely over the throat).
5. **Descend closed-loop on the MEASURED base plane** — `rig.descend_to_base`: 2.5 mm steps,
   re-measure the carried rim after each, recompute from the fresh `rel_z`:
   `eef_z = base + h_bowl − rel_z` (`rig.eef_for_base`). `rel_z` drifts 0.053 → 0.074 inside one
   carry, so an open-loop target is wrong by up to 2 cm; the loop tracked the commanded base to
   0.1–0.3 mm over 40 mm. **Never** write `base − rel_z`: that buries the bowl by exactly
   `h_bowl` = 49.5 mm and looks like a normal descent all the way down.
6. **Press 1.2 mm and score the contact test** (`rig.contact_test`): Δbase 0.5–0.7 mm short of
   Δeef = contact; Δbase ≈ Δeef = the bowl is over the throat, so go back to step 4 with a bigger
   `d`, **not** with another press.
7. Score from depth (base vs rim, tilt, lateral, source spot) and `finish` **held**.

Two measurement facts this run needed, both now library functions:

* `rig.robust_ring` — `rim_plane`'s world-z band fails on a tilt and `fit_ring3d`'s **+z seed
  collapses** on a bowl hanging at a pivot tilt (measured R 0.016 / tilt 41° against a true
  R 0.054 / tilt 12.6°). Brute-force the tilt first; it is 0 rows.
* `rig.measure_carried` — the carried object is achromatic (mean 145–148, i.e. inside
  `Scene.arm`) so colour cannot find it, and at a pitched tool the **hand block reaches down to
  `eef_z + 0.02`**, so a window with a generous top fits the hand's plate (R 0.07–0.14, long arc,
  0.7 mm residual — completely convincing). Window ladder + `|R − R_probe| < 2.5 mm` gate fixes
  it. The pad-exclusion cylinder must use the **measured** jaw offset, not `rig.JAW_DX/DY`.

## What does not work (all measured, do not re-derive)

| ending | what it gives | why |
| --- | --- | --- |
| **release** at the rim plane | −14.8 mm, 9.7° lean, centre 35.7 mm out | foot ring r ≈ 0.028 meets the 0.053–0.056**0** annulus in one point ~27 mm outboard of the COM → tips into the throat |
| **drop** 20 mm with the offset applied | 25.9° lean, −27.7 mm, destination shoved 36 mm | the impact rolls the foot over the annulus's inner lip. The single worst state |
| **nest** (centred) | −13 (partial) … −45 mm (full) | task 16's −13.2 mm was `False`: sinking in is not `on` |
| **hover** 9–11 mm clear, level, dead centre | +9.2 mm, `False` | clearance is not contact |
| **press with `d` ≈ 0.027** | base reaches the rim plane, **no contact** | the foot never touches the annulus; a press then just sinks the bowl |
| **tilted** approach (12–30°) | pads lose the rim, base edge slides past the outside | a rim can only be pinched near upright; a bowl tilted α presents ~21°+α to the jaws |

The lower bowl's rim **annulus is 0.053–0.0560** wide and the upper bowl's **foot ring r ≈ 0.028**
on this asset — with the COM 27 mm inside that contact, so nothing about the *released* geometry
can be fixed by aiming. That is why the ending is held, and why it is the held state the clause
accepts.

## Metrology on this scene (reproduced to 0.2 mm)

`rig.rim_plane` on a **2 mm** band under the rim's max: a 6 mm band adds the rim's shoulder and
biases the plane **2.6 mm low** (a whole tolerance). Then calibrate the height on the draw:
`h_bowl = median(zc − table_z)`, scoring `base = upper_zc − h_bowl` against the destination's own
`zc`. The often-quoted **0.0513** is `rim_z_max − table_z`; mixing the two is a 1.4 mm error.

| quantity | #4's draw | #5's draw (the pass) |
| --- | --- | --- |
| table_z | 0.9498 | 0.9498 |
| three bowls `zc` (rim plane) | 0.9995/0.9995/0.9996 | 0.99933/0.99932/0.99925 (0.1 mm spread) |
| `R` | 0.0541/0.0554/0.0544 | 0.0546/0.0545/0.0546 |
| `h_bowl` (zc − table) | 0.0497 | **0.04952** |
| annulus (top 1.5 mm radii) | 0.053–0.0566 | 0.0535–0.0560 |
| pad bottom − `eef_z` | 0.0425 | **0.0407** (flat at 0.030/0.045/0.060) |
| jaw-pair centre − eef | (−0.0201, −0.0117) | **(−0.0204, −0.0123)** |
| hand lead / its low edge | 0.1118 / `eef+0.088` | 0.1109 / `eef+0.087` |
| `gq` | 0.0031 (`-y` rim), 0.0025 (`+y`) | 0.0043 (`-y`), decaying to 0.0021 while held |

The last row matters: **`gq` is not a hold indicator on this task** — 0.0021 is the
"closed on nothing" band, and a bowl hung safely in the jaws with it for the whole 40 mm descent.
Judge a hold by the source spot (`object_gone`) and the picture.

Standing bowls are `R` 0.045–0.065 *and* 3–7 cm above the table; the plate has `R ≈ 0.065` at only
1.9 cm, so the two tests together keep it out. A blob 8 cm tall is a bowl on its side.

## Grasp mechanics that matter here

* **Pinch the `-y` rim, not the `+y` one.** With `side=-1` the bowl hangs at
  `eef + (−0.020, +0.038)` and the eef stays on the reachable side of the table; the `+y` rim puts
  the place pose ~6 cm further out, into the stretched-arm `-x` dead zone (`SKILL.md` §2).
  That hang offset is also the `hint` the carried-object mask needs — **get the sign right**: the
  wrong sign returns a 300–500 px fragment 4–9 cm away with `R` 0.017–0.020, and feeding that to a
  levelling rotation spins the tool 50–75° and drags the scene (task 17 lost its pair that way).
* `jaw = bowl + (0, −(R − 0.004))`, `eef = jaw − (JAW_DX, JAW_DY)`, `eef_z = rim − 0.055`
  (pad bottom ~12 mm below the lip), then `close_grip` until the jaws settle.
* **A rim can only be pinched near upright**: the wall is a ~21° cone, so a bowl tilted α presents
  ~21°+α and at 26° the fingers slide through — the tell is `gq` decaying
  `0.0395 → 0.0067 → 0.0042 → 0.0030 → 0.0019` with **no plateau**. That is a wedge slip, not a
  miss; do not re-aim the same pose. An **inverted** rim is the easy pinch (its wall faces 21°
  *above* horizontal), and jamming the object against something before closing also works.
* **The bowl slips in a lip pinch on every descent** (`rim − eef` went 0.0597 → 0.0555 over
  0.22 m, and 0.0633 → 0.0524 in one 6 mm step at contact). A falling fitted `R` (0.0538 → 0.0514)
  is the cheap slip detector. Re-measure after every descent — that is why step 5 exists.
* `gq ≤ 0.0022` after a close means **you are not at the object** (task 17 #2 closed 23 cm above a
  bowl and settled at 0.0015). Size descents from the row rate, not from a row cap.

## Identity: the bowls are indistinguishable, and a low release swaps them silently

The three bowls agree to 0.4 mm in rim z and 0.3 mm in `R`, and they read achromatic (mean
145–148, chroma ≤ 10) — **inside `Scene.arm`'s own class**, so `objects_on_table` returns 200–450 px
fragments of a 1 000–2 600 px bowl and any "bowl-coloured" picker finds nothing at all. Identify by
**x order + a position ledger**; measure a carried bowl by **geometry** only.

Task 17 #1 lost the whole episode to this, not to physics: a release driven down to table height
over the back bowl shoved that bowl **11 cm away** and the released bowl took its spot. Every later
stage then worked perfectly — re-grasp, carry, a clean centred placement — **on the wrong pair**,
and `finish` reported `on(bowl_2, bowl_3)` False. Nothing in the depth flagged it. So:

1. **Never drive the carried object's base below `lower_rim − 0.051`** over the target. Abort and
   re-plan instead. That one guard would have saved task 17's last 400 rows.
2. **Re-identify the destination at the last hover, before the final descent** — its rim is
   invisible once the carried bowl is within ~5 mm of it. `dest_now` does it: `rim_plane` in the
   ledger's window, reject the fit if it moved > 12 mm or changed `zc` by > 4 mm, else keep the
   ledger value. On this pass it repeated `zc` to 0.2 mm over 13 steps.
3. **Pick the destination by distance to the ledger, never by a measured height**: the front and
   back bowls differed by **0.1 mm** in `zc` and 22 cm in x, and sorting the standing candidates by
   rim z once carried the payload 26 cm to the wrong bowl.
4. **A released bowl can be scooped and carried** — riding on the hand block as a collar around the
   wrist, or clamped rim-to-rim with another bowl (`gq` 0.0048), following the arm. Any "object on
   the hand" scan must exclude the arm's *dark* parts (`mean < 70`, rgb 42–55) or its bbox drifts
   ~1 cm per move.
5. After any collision or unexpected end, run `look`/`show_objects` (0 rows) and re-derive which
   bowl is which **before touching anything else**.

## Budget

| episode | rows | outcome |
| --- | --- | --- |
| task 16 | 639/1000 | fail: ~200 on the stretched-arm `-x` dead zone, 40 on a picked forearm, and the wrong height |
| task 17 #1 | 929/1000 | fail on the **pair** (a low release swapped the bowls, then ~170 rows stacked the wrong two) |
| task 17 #2 | 261/1000 | fail on the **state**: held 9–11 mm above, dead centre, no contact |
| task 17 #3 | 604/1000 | fail on the **release**: reproduced the hover-and-drop, −27.7 mm / 25.9° |
| task 17 #4 | 990/1000 | ran out of rows; produced the −0.3 mm held state and the release that tipped it |
| **task 17 #5** | **272/1000** | **PASS, held at −0.75 mm** |

A clean run is **~270 rows**: 0 probe, 46 grasp, ~40 level + carry + hover, ~90 closed-loop
descent (16 depth calls are free), ~10 aim/nudge, 7 press + contact test, 0 score. Leave the rest
in reserve: a bowl that lands badly costs ~70 rows to re-grasp, and the whole point of the loop in
step 5 is that it can be re-entered from any pose where the bowl is still in the jaws.
