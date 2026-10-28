"""Skeleton. Copy to code/<name>/controller.py.

main(ctx) runs with the round already open. Keep this ctx. Do not call score,
reset, run, or submit, and do not open another connection.
"""
import arm


def main(ctx):
    obs = arm.observe(ctx)
    if obs.get("episode_ended") or obs.get("remaining_episode_steps", 0) <= 0:
        return
    arm.step(ctx, [arm.pack(obs)])
