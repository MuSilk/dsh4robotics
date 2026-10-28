"""Skeleton controller. Copy to code/<name>/controller.py and replace the body.

main(ctx) is called with the round already open. Use ctx, or pass that same
object into rig.goto / rig.solve_delta. Do not call rig.connect, score, reset,
run, or submit.
"""
import json


def _payload(result):
    return json.loads(result["content"][0]["text"])


def main(ctx):
    obs = _payload(ctx.call("libero_observe"))
    if obs.get("episode_ended") or obs.get("remaining_episode_steps", 0) <= 0:
        return
    _payload(ctx.call("libero_step", actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]))
