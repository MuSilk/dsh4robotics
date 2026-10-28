"""Load LIBERO-90 task 0 and step the offscreen environment once."""

import os

# Headless rendering on the shared A100. GPU 0 is the one with free memory.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task_id = 0
    task = suite.get_task(task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print(f"suite=libero_90 n_tasks={suite.n_tasks}")
    print(f"task_id={task_id}")
    print(f"name={task.name}")
    print(f"language={task.language}")
    print(f"bddl={bddl}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    env.reset()
    init_states = suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[0])
    action = [0.0] * 7
    obs, reward, done, info = env.step(action)
    image = obs.get("agentview_image")
    shape = None if image is None else tuple(image.shape)
    print(f"init_states={len(init_states)}")
    print(f"reward={reward} done={done}")
    print(f"agentview_image={shape}")
    print(f"success={bool(env.check_success())}")
    env.close()
    print("libero90-task0-ok")


if __name__ == "__main__":
    main()
