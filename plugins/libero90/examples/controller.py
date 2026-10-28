"""Example controller. Copy this file to code/<project>/controller.py.

Run it on the episode that is already open. Returning leaves the arm
where it is. Do not call register or finish from here.
libero_depth is allowed. It returns the agentview depth map.
"""
import json


def _payload(result):
    return json.loads(result["content"][0]["text"])


def main(ctx):
    obs = _payload(ctx.call("libero_observe"))
    print("instruction", obs["instruction"], "step", obs["step_id"])
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    step = _payload(ctx.call("libero_step", actions=[action]))
    print("step_id", step["step_id"], "ended", step["episode_ended"])
