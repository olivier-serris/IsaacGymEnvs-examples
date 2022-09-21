import isaacgym  # importing isaac_gym before torch is mandatory.
from omegaconf import OmegaConf
from isaac_gym_manipulation.envs import task_map
from omegaconf import OmegaConf
from isaacgym.torch_utils import to_torch
import hydra
import os
import torch
import time


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="main_config.yaml")
def main_loop(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    n_budget_step = 1_000_000
    env_class = task_map[cfg["env_name"]]
    grasp_env = env_class(
        cfg,
        sim_device="cuda" if cfg["sim"]["use_gpu_pipeline"] else "cpu",
        graphics_device_id=0,
        headless=cfg["headless"],
    )
    grasp_env.reset()
    n_step_done = 0
    start = time.time()
    while n_step_done < n_budget_step:
        n_step_done += grasp_env.num_envs
        # actions = np.random.uniform(-1, 1, )
        actions = torch.rand(
            (grasp_env.num_envs, grasp_env.num_actions), device=grasp_env.rl_device
        )
        observation, reward, done, info = grasp_env.step(actions)
        if n_step_done % (grasp_env.num_envs * 100) == 0:
            elapsed = time.time() - start
            step_per_seconds = n_step_done / elapsed
            print(f"step_per_seconds : {step_per_seconds}")
            print(f"step_per_seconds : {step_per_seconds}")


if __name__ == "__main__":
    main_loop()
