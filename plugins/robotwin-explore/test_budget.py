"""Pure checks for the RoboTwin exploration budget. No simulator."""
import unittest
from pathlib import Path

from worker import (
    AGENT_HIDDEN,
    CONTROLLER_HIDDEN,
    ROOT,
    action_width,
    arm_dims,
    controller_command,
    episode_cap,
    explore_base,
    gripper_indexes,
    instruction_text,
    load_step_limits,
    load_task_names,
    redact,
    validate_actions,
)


class BudgetTest(unittest.TestCase):
    def test_task_list_matches_the_step_limits(self):
        root = Path(ROOT)
        names = load_task_names(root / "env_cfg" / "eval" / "all_tasks.yml")
        limits = load_step_limits(root / "env_cfg" / "task_config" / "_eval_step_limit.yml")
        self.assertIn("adjust_bottle", names)
        self.assertEqual(len(names), 50)
        self.assertEqual(limits["adjust_bottle"], 400)
        self.assertTrue(set(names) <= set(limits))

    def test_explore_window_starts_after_the_eval_scan(self):
        self.assertEqual(explore_base(100000, 5), 100250)
        self.assertEqual(explore_base(100000, 1), 100050)

    def test_episode_cap(self):
        self.assertEqual(episode_cap(400, 1200, 0), 400)
        self.assertEqual(episode_cap(400, 1200, 1100), 100)
        with self.assertRaises(RuntimeError):
            episode_cap(400, 1200, 1200)

    def test_ee_row_and_gripper(self):
        self.assertEqual(action_width("ee", 6, 6), 16)
        self.assertEqual(gripper_indexes("ee", 6, 6), (7, 15))
        row = [0.1, -0.2, 0.9, 1, 0, 0, 0, 1, -0.1, -0.2, 0.9, 1, 0, 0, 0, 0]
        self.assertEqual(validate_actions([row], 16, (7, 15)), [row])
        bad = [0.1] * 16
        bad[7] = 2
        with self.assertRaises(ValueError):
            validate_actions([bad], 16, (7, 15))

    def test_qpos_width(self):
        left, right, name = arm_dims(ROOT, "demo_clean")
        self.assertEqual((left, right, name), (6, 6, "aloha-agilex"))
        self.assertEqual(action_width("qpos", left, right), 14)
        self.assertEqual(gripper_indexes("qpos", left, right), (6, 13))

    def test_instruction_without_expert_info_is_the_summary(self):
        text = instruction_text(
            "adjust_bottle",
            None,
            "seen",
            100000,
            Path(ROOT) / "description" / "task_instruction",
        )
        self.assertIn("bottle", text.lower())
        self.assertNotIn("{A}", text)

    def test_agent_payload_drops_the_seed_and_keeps_the_budget(self):
        hidden = redact({
            "seed": 100250,
            "budget_remaining": 10,
            "observation": {"seed": 100250, "step_id": 1},
            "step_id": 1,
        }, AGENT_HIDDEN)
        self.assertEqual(hidden, {
            "budget_remaining": 10,
            "observation": {"step_id": 1},
            "step_id": 1,
        })

    def test_controller_payload_drops_the_budget_too(self):
        hidden = redact({
            "seed": 3,
            "budget_remaining": 10,
            "observation": {"step_id": 1, "budget_used": 9},
        }, CONTROLLER_HIDDEN)
        self.assertEqual(hidden, {"observation": {"step_id": 1}})

    def test_all_tasks_hides_the_task_name(self):
        hidden = redact({
            "task_name": "adjust_bottle",
            "language": "Lift the bottle.",
        }, AGENT_HIDDEN | {"task_name"})
        self.assertEqual(hidden, {"language": "Lift the bottle."})

    def test_sandbox_does_not_mount_the_benchmark(self):
        command = controller_command(
            python="/data/changzou/miniconda3/envs/robotwin/bin/python",
            runtime="/data/changzou/deepseek4robotics/plugins/robotwin-explore/api/runtime.py",
            project="/tmp/code/policy",
            code_root="/tmp/code",
            socket_path="/tmp/sock/mcp.sock",
        )
        text = " ".join(command)
        self.assertIn("--unshare-net", command)
        self.assertIn("ROBOTWIN_SOCKET", command)
        self.assertIn("/tmp/sock/imajev.sock", command)
        self.assertNotIn("third_party/RoboTwin", text)
        self.assertNotIn("embodiments", text)


if __name__ == "__main__":
    unittest.main()
