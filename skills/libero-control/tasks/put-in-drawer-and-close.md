# Put an object into a drawer **and shut it**

`task.language` such as "put the butter at the back in the top drawer of the cabinet **and
close it**" (libero_90 tasks 3 and 4). Two stages, **in this order**: place the object in the
already-open drawer, verify it is in, then push the drawer shut. Closing first destroys the
goal, and reopening a drawer you just closed is expensive and risky.

**Scene frame first (0 rows, `SKILL.md` §1).** `rig.drawer_axis(s)` gives the bank's axis, the
**OUT sign** `S` (drawers open toward the camera side of the fronts' plane) and the fronts'
plane. KITCHEN_SCENE2's chest faces `S = +1` (+y out); KITCHEN_SCENE4's faces `S = −1` (−y out,
so every `+y`/`−y` in this file mirrors). Write the relations in `u = S · coord[axis]` and use
`front_face_on` / `handle_bars_on` / `panel_home_on` — the `+y`-only helpers return *plausible*
numbers 0.15–0.30 m out rather than failing.

The open drawer is a gift: this cabinet's top drawer starts ~0.15 m out, so there is no open
stage, and the free space is measured once offline from the opening depth call.

## 1. Identify the object before anything else

If the instruction carries a spatial qualifier ("at the front", "at the back"), it selects
*which* member of a pair, not where to put it — +x is "in front" (`SKILL.md` §1). Check the
pair is identical (tops within 1 mm, same footprint) and separated along that axis, and prefer
`rig.Scene.objects_on_table()`, which returns the blobs sorted by x.

**With no qualifier, the referent is the blob with no twin.** Task 5 had three packages on the
table and the named one was the odd one — dark brown (70, 49, 40), 8.1 × 4.9 × **2.8** cm — while
the other two were an identical butter pair (7.6 × 3.9 × **1.8** cm, (103, 54, 31), tops within
1 mm, the same asset as task 4's butter). Compare colour **and** height; the pair matches to
1 mm and to within ~3 in RGB. Grasping the wrong member of a twin pair costs the episode even
though every later stage looks perfect.

Two traps, both measured:

* a third, darker bar may sit near the pair and **not** be the object named. On task 3 it was
  3 cm away; on task 4 it was a 7.7 cm bar whose flat top stood 1 cm higher (0.978 vs 0.9677)
  and whose colour was flatter (93, 67, 51) than the butters' (116, 61, 33).
* the pair's x ranges can **touch** (3 mm apart on task 4), so a "has it gone" box of ±0.06
  around one still catches the other's corner — and reported the butter still present after a
  *correct* lift. Use ±0.04, or segment both blobs and match centres.

## 2. Measure the box (0 rows)

One `libero_depth` gives the interior. The plane list alone is **not** enough — and not because
a near wall hides the floor: on task 5 the `z = 1.115` floor plane claimed y −0.244…−0.102 while
the columns showed free floor only over y −0.205…−0.115, because the drawer's **back wall has a
3.4 cm wide top face at 1.175** that the plane bucket happily includes. Run the free-column test
over the mouth — for a grid of (x, y) print the max z of a 12–15 mm column:

* a column reading the **floor** is free; the **rim** is a wall; the **table** z means the
  sight line passed under the box (not that the box is absent);
* task 4: floor 1.113 over x −0.10…0.00, y −0.21…−0.11, with the counter top at 1.175 only
  behind y −0.229 — i.e. the box is *not* roofed, so a top-down drop works. Task 5:
  x −0.105…+0.015, y −0.205…−0.115 — a *smaller* window than the plane suggested, and a drop
  point taken from the plane's extent would have landed on the back wall;
* also confirm the drawer is actually out: its own panel plane must be present.

## 3. Grasp: aim the pad *bottom*, not the eef

A flat object breaks the usual rule: 1.8 cm tall means the pad bottom sits ~6 mm above the
table, so `eef_z = table_z + 0.006 − PAD_DZ` — on task 4, **0.9132** with the table at 0.9498,
i.e. the eef reference 3.7 cm *under* the table plane. That is legal (`SKILL.md` §3).

Straddle the object's narrow dimension (the jaws close along y), centre the **jaw pair** on the
top-face centroid, descend straight down, and expect the jaws to settle at `gripper_qpos` ≈
half the object's width — **0.0195** for a 4.4 cm butter, 0.0195 for a 4.3 cm stick, which is a
real pinch, not "closed on nothing". Then lift ~0.11 m and verify the old spot is empty.

**A box taller than the ~2.2 cm pad band** (task 5: 2.8 cm) needs the pad bottom at
`table_z + 0.013`, i.e. `eef_z = table_z + 0.013 − PAD_DZ` = **0.9225**, so the band covers its
top 1.5 cm and the tips stay 1.3 cm off the table; the jaws then settled at 0.0240 on a 4.9 cm
side.

**Before that descent, check the hand clears the cabinet's handle bar** (`SKILL.md` §3:
`eef_y − 0.111 > bar_front_y + 0.01`). This is the trap that cost task 5 ~215 of its 278 grasp
rows: a straight
descent at the object's own `eef_y` stood the hand's front corner on the bar's top face
(`eef_z + 0.082` = 1.1469) and every axis went dead in a way that reads exactly like a freeze.
If the object is within ~11 cm of the cabinet, descend at `eef_y ≥ bar_front_y + 0.131`, drop
until the hand's **top** (`eef_z + 0.13`) is below the bar's bottom (0.990 here), slide in `−y`
at that height, then descend the last few centimetres.

## 4. Place: go deep, then let the last centimetres fall

* Carry above the rim (`eef_z ≥ rim + 0.09`) so the object clears the panel top, and leave room
  for its **long** axis in x (task 4: a 7 cm butter gripped at x −0.05 spans −0.085…−0.015
  inside a mouth of −0.128…+0.008).
* Put the drop point at least 5 cm behind the panel's inner face (task 4: y −0.170, i.e. 9.5 cm
  behind). This is the whole trick that makes `tasks/put-in-drawer.md`'s "the gripper cannot
  descend into the drawer" problem disappear: with the drop point deep, the finger bottoms are
  *inside* the box, so nothing lands on the panel's top face and no yaw workaround is needed.
* Lower until the servo stalls. **A stall ~2–3 cm above the floor is the hand meeting the box
  mouth, not a failure**: task 4 stopped at eef_z 1.0957 (1.7 cm up). Release there with a small
  `+z` bias on the opening rows so the pads clear, and let the last 2.5 cm land flat.

## 5. Verify **before** you close

Once the drawer is shut `libero_depth` cannot see inside, so this is the last observation you
get:

```
col = max z of scene points in a 12 mm column at the drop point   # gripper OUT of the column
on_the_floor = |col - (drawer_floor_z + object_height)| < 5 mm
```

Task 4 read **1.1317** against a prediction of 1.1309 (floor 1.113 + butter 0.0179); task 5
read **1.142** against 1.1411 (floor 1.113 + 2.8 cm pudding). Also
confirm the object has left its old table spot (`zmax` collapse, then the picture). Do not use a
table-level blob search around the box: it also finds the *lower* drawer's handle.

Closing does not move an object sitting on the drawer floor — the box interior is 6.2 cm tall
and the objects 1.8–2.8 cm, so the rim cannot catch it.

## 6. Then close, exactly as `tasks/close-drawer.md` says

The object changes nothing about the push: same handle bar, same panel band, same stall rule.
The one thing to carry over is where the **pushing surface** is: the gripper's hand block, 0.111 m
ahead of `eef_pos` (`SKILL.md` §3, §7). Aim it at the bar with
`z_target = bar_bottom − 0.094` (**1.045** here, not the 1.090 that aims the *pads* at the panel
band) and call `rig.push_to_stop(..., ref_y=ref_y)`; it stops on the eef's stall, proves the
freeze with one `+z` row, and takes the 1 mm verdict against the sibling closed drawer's panel.

Task 4 died twice on exactly this: a push at pad height stalled with the panel 2–4 mm proud
while every picture said "shut", and the `Close` predicate failed. The push at bar height took
the same drawer from −0.2248 to −0.2305 in 7 travel rows.

**Do not let `home=False` alone stop you.** The reference is the *other* drawer's panel, and on
task 5 the pushed top panel came home at −0.2305 while the closed sibling's sat at −0.2289 —
1.6 mm, so `panel_home` said False on a drawer that had stalled on its joint and that the
episode scored as closed. Pass `floor_z=<measured floor>` to `push_to_stop` and it adds the
decisive signal: an open drawer's floor plane (z 1.113–1.115, n ≈ 5 600) is **gone** once the
drawer is shut, and no panel-plane ambiguity can fake that. If `home=False` while the floor
plane has vanished and the rim plane's front edge has moved to −0.228, the drawer is home.

## Budget

Task 3 did the whole thing in **159 of 600** rows with no re-runs: 0 probe, 47 grasp, 75 place
(45 the carry), 37 close. Task 4 passed in **241**: 0 probe, 65 grasp, 105 place, 45 on a close
at the wrong height, 6 to retreat and re-measure, 20 on the close that worked. Task 5 (2.8 cm
box, object within 11 cm of the cabinet) took **407**: **278 grasp**, of which ~215 were the two
straight descents onto the handle bar and the diagnostics that went with them — the grasp that
worked was 31 rows — then 82 place, 47 close. Plan **~40 rows per 0.4 m of
carry** — the carry dominates — and treat a second attempt as evidence you mis-measured
something, not as a normal cost.
