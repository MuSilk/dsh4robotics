# Move the bowl

Grasp the black bowl and put it where the instruction says. Measure the bowl and that place with `libero_depth` for this placement. Do not reuse a coordinate from another episode.

**If the instruction says "stack … on …" and the target is another bowl, read
`tasks/stack-bowl.md` instead of this file for the placement half.** Here the destination is a
surface and "does it sit there" is answered by a plane; there the destination is another object
and the verdict is a **height** clause (`on(upper, lower)`) that a bowl sunk into the other
bowl's mouth fails even though it looks stacked. The grasp below is the same.

The shared method, cameras, and depth correction are in `SKILL.md`.

If the instruction also closes a drawer, do `tasks/close-drawer.md` first and retract clear of the cabinet before touching the bowl. The bowl sits on the table just in front of the open top drawer, beside its handle, and the counter is directly behind that handle. Closing first keeps the bowl out of the push. Placing the bowl first leaves it on the counter in the middle of the push, and the wrist at rim height knocks it off.

During the episode the tools do not say whether the instruction is satisfied. A closed drawer alone does not finish an instruction that also names the bowl. Call `finish` when the scene matches the instruction. `finish` and a step that hits the budget then report `task_complete`.

## The bowl

It is the speckled dark-grey bowl on the table in front of the open drawer, not a glossy black object. Its rim is the highest horizontal band of its blob: take the rim's z from the top decile of the blob's z, its outer edge as the extreme along y, and its centre from the *bounding box* of those pixels (the median is pulled toward the far rim, which is the part the depth sees most of). **Better, fit a circle to those top-decile pixels** — `rig.Scene.rim_circle` does — because the blob's bbox also swallows the arm's shadow and the drawer's dark handle when they are nearby: on one placement the blob centre came out (−0.129, −0.006) against a fitted rim centre of (−0.102, −0.016), a 2.5 cm error in the number the grasp is built from. Three placements: rim 4–5 cm above the table, 0.09–0.11 m across (fitted R = 0.051 on the last one), body down to `z ≈ table + 0.008`. It is wider than the 0.068 m the jaws open, which is the whole reason for the rim pinch, and a 0.02 m spread in diameter is 0.02 m of error in where the outer pad lands.

Grasp it on the rim, not on the body. The gripper opens to about 0.068 m and the bowl is wider.

`rig.grasp_rim(ctx, rim, q_tgt)` does the whole grasp from a `rig.Scene.rim_circle` dict —
jaw-pair offset, pad-band height, settle-close and lift — and `rig.rim_grasp_target(rim)`
prints the target first so it can be checked against the depth before any row is spent.
The notes below are *why* its constants are what they are.

* **Put the JAW PAIR on the rim wall, not `eef_pos`.** `eef_pos` sits about
  (−0.014, −0.015) m from the middle of the fingers with the wrist down, so a target built
  as `eef_y = rim_outer_y` puts the jaws 1.5 cm inside or outside the wall — more than the
  wall is thick, and the jaws close past it. Aim the pair centre at the rim's outer wall,
  so `eef_y = rim_outer_y + 0.015` (and `eef_x = rim_centre_x + 0.014`). `rig.Scene.jaw()`
  measures the offset; SKILL.md §3 has the code.
  The rim edge facing away from the cabinet is the freest; on the placements seen that is
  +y. With the jaws open ~0.07 m one pad lands well outside the rim and the other inside
  the mouth — that is the pinch you want, so centre the pair on the rim's *outer* edge
  rather than on the bowl's body or on its bounding-box centre.
* **Put the pad bottom just below the rim's top lip, and take the rim z from depth**, never
  from where the rim looks high in the picture, because image rows are mostly world x.
  The pads are a couple of cm tall and their bottom is `eef_z + 0.043`, so
  **`eef_z ≈ rim_top_z − 0.055`** (with `rim_top_z` = the max z of the rim ring). A
  successful pinch on the next session: rim top 1.0008, close pose `eef_z` 0.941, pad
  bottom ≈ 0.985. The old formula here, `rim_z − pad_offset − 0.02`, puts the pad bottom
  2 cm *below* the lip on the bowl's sloped side, where it slips.
* **Then close until the jaws settle — 16–18 rows, not "a few".** The jaws move only
  ~0.0023 m/row, so 4 close rows leave them 6 cm open and the lift simply pulls the pads
  past the rim. Hold `[0,0,0,0,0,0,+1]` and watch `gripper_qpos` stop changing
  (`rig.close_grip`). This single mistake cost ~280 rows on one episode across six
  "failed" grasps, every one of them aimed correctly.
* The bowl's bottom then tracks the eef (measured: bottom ≈ `eef_z` to `eef_z` + 0.02 m).
  Set it down by descending until `eef_z ≈ support_top_z − 0.005`, then release.
  Keep `gripper = +1` in every row of the carry; a zero row reopens the gripper.

**Never test the pinch with `gripper_qpos`.** A rim is a thin wall, so a *good* pinch
reads ~±0.004 — indistinguishable from jaws closed on nothing (±0.002), and a check like
`abs(qpos) < 0.008` calls a perfect grasp a failure. Two better tests, in order:

1. **Look at the picture** (`rig.save_frame`, 0 rows). The bowl hanging in the jaws is
   unmistakable, and it is the only check that was right every time.
2. Re-read depth over the old spot. A **brightness mask is not enough**: the wood table
   grain falls inside the same grey window (mean 75–178) and reported "bowl still there"
   on a correct grasp. Require that the blob's `zmax` has collapsed below ~table + 0.03
   *and* that its extent no longer matches the bowl.

If it is still on the table, the pinch was off — re-measure the rim, and shift by the
difference rather than repeating the same numbers.
