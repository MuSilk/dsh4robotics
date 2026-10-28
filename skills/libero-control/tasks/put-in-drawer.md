# Put an object into an open drawer

The drawer is **already pulled out** — the job is a rim grasp plus a carry *into* the box, with
no push stage. Do not follow `close-drawer.md` here: closing the drawer destroys the goal.
**Unless `task.language` ends with "and close it"** — then the goal is "in the drawer" *and*
"drawer shut", so read `tasks/put-in-drawer-and-close.md` and do the close at the end.

**Scene frame first (0 rows, `SKILL.md` §1).** `rig.drawer_axis(s)` gives the bank's axis, the
**OUT sign** `S` (drawers open toward the camera side of the fronts' plane) and the fronts'
plane. KITCHEN_SCENE2's chest faces `S = +1` (+y out); KITCHEN_SCENE4's faces `S = −1` (−y out,
so every `+y`/`−y` in this file mirrors). Write the relations in `u = S · coord[axis]` and use
`front_face_on` / `handle_bars_on` / `panel_home_on` — the `+y`-only helpers return *plausible*
numbers 0.15–0.30 m out rather than failing.

## The reach problem is a placement problem

The "the gripper cannot descend into the drawer" warning below is real only if you drop the
object near the front of the box. Put the drop point ≥ 5 cm behind the panel's inner face
instead and the finger bottoms stay *inside* the box: on libero_90 task 3 a drop at
y = −0.18 (box interior y −0.235…−0.107) let the descent reach eef_z 1.096 — the object's
bottom 26 mm above the 1.113 floor — with no finger anywhere near the panel top. If the
descent stops ~2.5–3 cm high anyway, that is the hand meeting the mouth: release there, the
last drop lands flat (verify `column_max_z` = floor + object height), and no yaw is needed.
Also leave room for the object's *long* axis in x, and take the column test with the gripper
clear of the column.

## Two reach problems BEFORE the box

* **Which rim to pinch is a scene measurement, not a preference** (SKILL §0.4d). The hand is a
  ~0.2 m plate centred on the eef — 11 cm on the finger side, 9 cm on the other, at
  `eef_z+0.07…+0.09` — so every rim has two pinches (plate OUT past it, or IN over the object)
  and four rims, and a blocked side costs a descent. On libero_90 task 24 the §4b default (−y
  rim, plate OUT) was blocked by leaning cutting boards **3.5 cm** behind the bowl's rim: two
  descents stood the hand's plate on their top edge (`eef_z` 1.1149, then 0.9923, `+z` alive,
  `‖rot_err‖` 0.045, a steady +x drift, nothing within 4 cm of the pads). The +y rim was blocked
  the other way — by the **open drawer's own front panel** (top 1.033 against the plate's 1.024).
  What worked was the **+x rim with the wrist yawed +90°**:
  `rig.rim_grasp_axis(rim, (1,0,0), hand_out=True)`, where the table is bare. Read all four
  sides in the probe (0 rows), never after the first stall.
* **The descent into the box can die on ARM REACH, not on the box.** At (x +0.02, y +0.16) this
  arm cannot get below `eef_z ≈ 1.11` — nothing in the depth under the payload, `+z` alive,
  `‖rot_err‖` 0.02, the same short-reach jam as SKILL §2 (task 21's was (+0.03, 0.10, 1.25)).
  The reachable direction is **−x**, so aim the payload 1.5–2 cm off the mouth's centre toward
  −x instead of releasing 10 cm up, and remember a yawed pinch hangs the payload ~7 cm off the
  eef: carry with the MEASURED `d_xy` (`rig.measure_held`), never with `R`.

## Measure the box, not just the face

One `libero_depth` gives everything. What you need is the **interior**:

* floor plane z, wall-top / rim z, front-panel **inner** face y and top-face z, back-wall inner
  face y, both side-wall x offsets;
* then the **free-column test** for the mouth: for a grid of (x, y) over the box, print the max
  z of every scene point in the column. Columns with nothing above the floor are where the
  object can go down. On the task-2 cabinet this showed an opening y −0.225…−0.098 with the
  panel's top face (z 1.172) blocking in front and the back wall behind — a *plane* reading
  alone would have suggested a counter overhanging the whole mouth.
* A near side wall hides part of the floor: the floor plane's x extent can be much narrower
  than the box. Trust the wall planes, and remember that a column reading the *table* z means
  the sight line passed under the box, not that the box is absent.

* **The visible floor is an inner bound, not the mouth.** On task 24 the floor spanned
  x −0.130…+0.022 while the box's inner walls stood at **+0.0805** and **−0.1315**
  (`rig.mouth_walls`, and note `~arm` must be OFF: the bright interior is inside `Scene.arm`),
  and its back wall at y **+0.2286**, i.e. PAST the fronts' plane — the box slides into the
  cabinet, so "between the panel and the fronts" is the wrong model. A 2R 0.110 bowl against the
  printed 0.143 × 0.120 "mouth" looked like 11 mm of margin; the real margin was 4 cm.

## The reach problem nobody warns you about

With the wrist down the finger bottoms sit at `eef_y − 0.061 … +0.036`. A rim grasp puts the
jaw pair ~0.047 m from the object's centre, so the **outer finger reaches ~0.09–0.098 m in
front of the object's centre**, while the object is only ~0.05 m in radius. An open drawer is
typically 0.135–0.16 m deep, so:

> **The gripper usually cannot descend into the drawer.** The outer finger lands on the front
> panel's top face (0.5–1 cm above the rim) and stops the descent.

Two fixes, both used or costed on task 2:

1. **Drop the last few centimetres** (costs no rows). Place the object so it is fully inside
   the interior with ~2.5 cm clearance to every wall, descend only until the pad bottom is
   ~8 mm above the panel top (`eef_z = panel_top − PAD_DZ + 0.008`), release, and let it fall
   the last ~25–30 mm. Measured on a 0.10 m bowl into a 0.135 m drawer: it landed upright, its
   rim top reading floor + object height to 0.5 mm.
2. **Yaw the wrist 90°** so the jaws straddle along x (the drawer is usually much wider in x
   than deep in y) and set the object down with no drop. Costs ~20 rotation rows plus a
   re-measured jaw offset for the new orientation.

## The window above the mouth is what limits the descent, not the mouth

The mouth is a *plan view*; what a wrist-down hand needs is **free air**, and a chest with closed
drawers above the open one has very little of it. Measure it (0 rows) as: `rig.box_free_columns`
over the mouth at 1–2 cm, plus the z band of every overhang that stands proud above it — on task 26
(libero_90, KITCHEN_SCENE4, bottom drawer) the bars of the two closed drawers stood at y +0.1851 in
bands **1.0666–1.0735** and **1.1397–1.147**, over a mouth whose panel inner face was y +0.070.

The hand is **15.7 cm tall** (`rig.hand_stack`: pads +0.043, a 1 cm bar +0.075…0.095, a 3 cm block
+0.10…0.14, a 7.6 × 20 cm box +0.135…0.20) and **20 cm long along the jaw axis**, so:

* an 11.5 cm free window **cannot** be threaded with the payload deep in the mouth: the wrist block
  rests on the upper bar and the deepest clean `eef_z` is `bar_bottom − 0.135…0.20`;
* at that pose the payload's bottom is `eef_z + 0.054 − r`, and the difference to the floor is a
  **drop**. A straight drop of ≤ 10–15 cm is fine if the payload's whole plan view is inside the
  walls and the release is symmetric (§Release) — task 24's bowl free-fell 15 cm and landed upright,
  and MuJoCo's default contact adds no restitution. Planning a drop is cheaper than 40 rows of a
  descent the geometry forbids;
* if the plan view is too big for the mouth, rotate the payload **while carrying** (a 15.6 cm bottle
  is 14.0 cm across at 44° against a 13.0 cm free depth, and 4.4 cm at 0°) — but rotate it at a
  parking pose: an open or loaded hand sweeps a 0.2 m disc (`SKILL.md` §3).

## When the object is not a rim

A drawer task says nothing about *how* to pick the thing up. If it is flat, straddle it; if it is
lying on its side and can roll, `tasks/grasp-cylinder.md` — the jaw axis must be perpendicular to its
own axis and the pad band **above** that axis, which is the opposite of every rim pinch in this file.

## Release

* Rows of `[0,0,0,0,0,0,−1]` are **not** a position hold: with the object still pinched the eef
  sank 1.1 cm in eight rows on task 2 (pad bottom ended 4 mm *below* the panel top). Servo the
  release pose in the first release row, or hold with `gripper=−1` plus a small +z bias.
* Retreat **+z first** until the pads clear the rim, then out along +y, then down.
* Verify from depth: the object's rim z should equal the drawer floor plus the object's own
  height measured on the table (that equality, not the pixel count, is the proof it is sitting
  on the drawer bottom and not on a wall or the rim), and its x/y must be inside the interior.
  Do it as a GRID, not one column: `rig.container_top_scan(s, mouth_xybox, floor_z, h_obj)`
  classifies every cell (at `floor + h_obj` = the object's top, at the floor = floor, above =
  a wall or the drawer-above's bar) and requires the cells' centroid inside the mouth. A rim fit
  is not a verdict inside a bright drawer — on task 24 the bowl and the interior merged into one
  4 000-pixel blob ("ring" R 0.075, residual 10 mm). And the drawer ABOVE overhangs the mouth:
  the closed one's bar stood at z 1.0670–1.0735 protruding to y +0.185, 3–4 cm above the open
  box's rim and inside its back half, exactly where the hand's plate arrives.
