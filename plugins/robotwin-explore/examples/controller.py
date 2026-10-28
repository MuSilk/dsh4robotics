"""Skeleton controller. Copy to code/<name>/controller.py and replace the body.

main(ctx) is called with the round already open. Use the ctx you are given.
Do not call score, reset, run, or submit. Do not open a second connection.
"""
import json


def _payload(result):
    return json.loads(result["content"][0]["text"])


def _row(obs):
    if obs.get("action_type") == "qpos":
        return obs["left_joint_pos"] + [obs["left_gripper"]] + obs["right_joint_pos"] + [obs["right_gripper"]]
    return obs["left_endpose"] + [obs["left_gripper"]] + obs["right_endpose"] + [obs["right_gripper"]]


def main(ctx):
    obs = _payload(ctx.call("robotwin_observe"))
    if obs.get("episode_ended") or obs.get("remaining_episode_steps", 0) <= 0:
        return
    _payload(ctx.call("robotwin_step", actions=[_row(obs)]))
