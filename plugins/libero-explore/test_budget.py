"""Pure checks for the exploration budget. No simulator."""
import unittest

from worker import (
    AGENT_HIDDEN,
    CONTROLLER_HIDDEN,
    all_episodes,
    all_state_ids,
    controller_command,
    episode_cap,
    parse_eval_inits,
    redact,
)


class BudgetTest(unittest.TestCase):
    def test_eval_covers_every_scene(self):
        self.assertEqual(all_state_ids(50), list(range(50)))

    def test_rejects_empty(self):
        with self.assertRaises(RuntimeError):
            all_state_ids(0)

    def test_all_episodes_cover_each_task(self):
        self.assertEqual(all_episodes([2, 1]), [(0, 0), (0, 1), (1, 0)])

    def test_all_episodes_limit_per_task(self):
        self.assertEqual(all_episodes([5, 2], 2), [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_all_episodes_limit_does_not_pass_the_end(self):
        self.assertEqual(all_episodes([2], 10), [(0, 0), (0, 1)])

    def test_parse_eval_inits(self):
        self.assertIsNone(parse_eval_inits("all"))
        self.assertIsNone(parse_eval_inits(""))
        self.assertEqual(parse_eval_inits("3"), 3)
        with self.assertRaises(RuntimeError):
            parse_eval_inits("0")
        with self.assertRaises(RuntimeError):
            parse_eval_inits("nope")

    def test_all_episodes_rejects_empty_task(self):
        with self.assertRaises(RuntimeError):
            all_episodes([2, 0])

    def test_suite_payload_drops_task_index(self):
        hidden = redact({
            "task_id": 3,
            "bddl": "/tmp/task.bddl",
            "instruction": "put the bowl on the plate",
            "task": {"task_id": 3, "language": "put the bowl on the plate", "bddl": "/tmp/task.bddl"},
        }, AGENT_HIDDEN | {"task_id", "bddl"})
        self.assertEqual(hidden, {
            "instruction": "put the bowl on the plate",
            "task": {"language": "put the bowl on the plate"},
        })

    def test_episode_cap(self):
        self.assertEqual(episode_cap(400, 2400, 0), 400)
        self.assertEqual(episode_cap(400, 2400, 2300), 100)
        with self.assertRaises(RuntimeError):
            episode_cap(400, 2400, 2400)

    def test_agent_payload_drops_scene_index_and_keeps_budget(self):
        hidden = redact({
            "init_state_id": 3,
            "budget_remaining": 10,
            "observation": {"init_state_id": 3, "step_id": 1},
            "step_id": 1,
        }, AGENT_HIDDEN)
        self.assertEqual(hidden, {
            "budget_remaining": 10,
            "observation": {"step_id": 1},
            "step_id": 1,
        })

    def test_controller_payload_drops_budget_too(self):
        hidden = redact({
            "init_state_id": 3,
            "budget_remaining": 10,
            "observation": {"step_id": 1, "budget_used": 9},
        }, CONTROLLER_HIDDEN)
        self.assertEqual(hidden, {"observation": {"step_id": 1}})

    def test_sandbox_hides_dataset(self):
        command = controller_command(
            python="/data/changzou/miniconda3/envs/ds4ro/bin/python",
            runtime="/data/changzou/deepseek4robotics/plugins/libero-explore/api/runtime.py",
            project="/tmp/code/policy",
            code_root="/tmp/code",
            socket_path="/tmp/sock/mcp.sock",
        )
        text = " ".join(command)
        self.assertIn("--unshare-net", command)
        self.assertIn("/tmp/sock/imajev.sock", command)
        self.assertNotIn("init_files", text)
        self.assertNotIn("third_party", text)


if __name__ == "__main__":
    unittest.main()
