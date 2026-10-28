# Grasp a lying round object (a bottle, a can, a rolling pin)

Use this when the object rests **on its side**: it rolls, it has no rim to pinch, and its axis
matters. It is the one grasp class where the pad band's *height relative to the object's axis*
decides success, and where a miss does not just fail — it **moves the object**.

Measured on libero_90 task 26 (a wine bottle lying on a table: length 0.156, diameter 0.045, axis
z 0.9722 on a 0.9498 table). The numbers transfer as *relations*; the diameters do not.

## The three numbers, from one depth call

1. **The axis**, from the object's **top ridge**: a resting cylinder's highest points lie directly
   over its axis, so the ridge's projection *is* the axis. Do not use the blob's centroid — only the
   upper band is visible and a colour mask clips one side, which measured **1 cm** of error on a
   4.5 cm bottle, i.e. most of a jaw's aim. `rig.axis_from_ridge(scene, mask)` returns
   `point`, `dir`, `angle`, `t`, `axis_z`.
2. **The diameter**, as `2 × (top − support)` — the top of the visible surface (`p95 z` of the
   class) minus the **local** support plane under the object, never the class's z extent: a colour
   class misses the shadowed lower side, so its `zmin` reads 1–2 cm high and the object looks as if
   it were floating.
3. **The yaw**: `yaw_quat(Q_DOWN, angle − 180)`. There are exactly two candidates 180° apart (same
   jaw-axis line, opposite hand), and **they are not equivalent**: on task 26 the `angle − 180`
   variant was clear and the `+0` variant put the palm's corner 2–7 cm through a leaning board.

## Audit the corridor before spending rows

`code/bottle26/audit.py` is the shape to copy: build a **chroma** height field of the scene
(`max−min > 30` — the robot is grey to within 2, so this removes it *exactly*, which `~Scene.arm`
does not: it leaks ~10 % of the arm's own pixels and one leaked pixel faked a 1.4 m obstacle), then
translate+rotate the **arm's measured silhouette** into each candidate pose and test every arm point
at its own z. Two contracts that decide whether the audit is honest:

* the silhouette is stored **relative to `eef_pos`**, so the candidate pose must be given as the
  **eef**, not as the jaw — the 2 cm difference flipped a verdict from "clear" to "81 points
  blocked" on task 26;
* the object itself is removed from the field (the plate may rest on it), but the *other* objects —
  boards, bowls, and the destination — must not be.

The audit is a knife edge along a bottle: on task 26 the fat middle (`t` −0.02…+0.02) put the palm's
corner through the boards, while `t ≤ −0.025` and `t ≥ +0.03` were clear. **Pick the grasp point
from the audit, not from the object's centre of mass.**

## Descend against the measured axis, not at a planned height

The pad band must end up **6–8 mm above the axis**: `eef_z = rig.band_for_axis(axis_z, 0.006)`
(pad band `eef_z+0.043…+0.065`). Below the axis the squeeze's contact normals point *up* and a free
cylinder is thrown out of the pinch — measured: a band 3 mm **low** closed to `gq 0.0028` (pads
meeting) and popped the bottle up into the fingers' hooked prongs, while the same pose with the
band high presses it into the table.

The loop that worked (and its cost — one depth read per 2 cm is free):

```
for each 2 cm slice:
    goto(xy, z - 0.02)            # jaws OPEN, quat servoed
    s = snapshot(); b = rig.axis_from_ridge(s, mask)      # 0 rows
    stop when  pad_bottom <= b["axis_z"] - 0.004
```

Two guards on it:

* **the yaw must be verified, not assumed.** `rotate_to` that hits `max_rows` returns a large
  residual (task 26: 0.60 rad, 34° short). A 34°-short yaw turns the pinch into a 12 cm projected
  width against a 6.8 cm opening: the pads closed *past* the object and the arm itself was pushed
  3.4 cm. Gate the descent on `‖rot_err‖ < 0.04`.
* **rotate away from the object.** With the wrist down the hand is 20 cm long along the jaw axis and
  15.7 cm tall (`rig.hand_stack`), so an in-place rotation sweeps a 0.2 m disc at `eef_z+0.04…+0.20`
  and shoves a lying bottle. Park ≥ one hand-length away, rotate, then translate.

Then `close_grip` until settled and **prove it in 5 rows**: lift 4–5 cm, one depth read, require the
object's *top* to have risen with the eef and `gq` to be off the "pads on nothing" band
(0.002–0.003). Task 26's tell that a grasp had failed was `top` unchanged to 0.0 mm with `gq` 0.0007.

## Recovery: a miss ploughs the object

A parallel-jaw close on a free object it did not centre **pushes it**: measured 5 cm of object and
3.4 cm of arm in one close, 24 cm over two — more than this arm's reach margin. So:

1. **Re-probe (0 rows) before re-aiming**, and rebuild the aim from the new pose. A plan that reuses
   the pre-failure pose is aiming at where the object used to be.
2. **Check the reach envelope before the second attempt.** On this rig the pad bottom bottoms out at
   **0.99 for x ≤ −0.40** and still reaches 0.958 at x −0.335, so an object dragged to x −0.50 is
   un-graspable, whatever the depth says. The tell is a descent that *rises* while commanded down
   with an xy residual that will not close (`SKILL.md` §3).
3. **Drag it back with the open jaws** — cheap per centimetre, expensive per attempt:

   * no rotation needed if the wrist is already roughly perpendicular: the hand's −x extent from the
     eef is ~0.05 (open jaws), so an eef at x −0.455 can press an object at x −0.50;
   * descend to the **lowest reachable** `eef_z` (pad bottom just under the object's top: 0.99 here),
     then push **+x** in 3-row saturated bursts at `gripper = −1`;
   * re-read the depth after every burst; measured 3 cm per burst of 18 rows, and the *whole* stage
     cost ~60 rows per burst because each iteration re-approached from 15 cm up. Descend once and
     push several bursts from the same pose.
4. If the object is still out of reach with ~25 % of the budget left, that is the point to decide
   the clause is lost — not at 1000/1000.

## The place: the mouth is not the free air

The object's plan view decides the rotation, and the free air above the mouth decides the height:

* rotate the payload **while carrying** so its long axis fits the mouth's *plan view* (a 15.6 cm
  bottle at 44° needs 14.0 cm of a 13.0 cm free depth; at 0° it needs 4.4 cm) — at a parking pose,
  for the sweep reason above;
* the pad band holds the payload at `eef_z + 0.054`, so its bottom is at `eef_z + 0.054 − r`; the
  deepest clean `eef_z` is set by the tallest overhang the hand must pass (`bar_bottom − 0.135…0.20`);
* release **symmetrically** (open both jaws at `dz = 0`) where the whole plan view is inside the
  walls: a straight drop of ≤ 10–15 cm costs nothing (MuJoCo adds no restitution) and beats 40 rows
  of a descent the geometry forbids.

## Failure table

| symptom | cause | fix |
| --- | --- | --- |
| `gq` settles 0.002–0.003, object still on the table, top z unchanged | the yaw was short (34° here) or the band was below the axis — the pads closed past it | verify `‖rot_err‖ < 0.04`; band `axis + 0.006` |
| the object pops **up** into the fingers | band centred below the axis | raise the band 6–8 mm above the axis |
| the object is 5–20 cm from where the plan says | the previous close ploughed it | re-probe; if outside the envelope, drag it back |
| the descent rises while commanded down | the target xy is outside the reach envelope | move toward the base until the xy closes |
| the palm stands on something behind the eef at `eef_z+0.08…0.20` | the 20 cm hand bar over clutter | pick a different `t` along the object (the audit says which) |
