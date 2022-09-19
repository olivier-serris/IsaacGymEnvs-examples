import isaacgym  # importing isaac_gym before torch is mandatory.
import os
from isaac_gym_manipulation.envs import task_map
import hydra
from omegaconf import OmegaConf
from isaacgym.torch_utils import to_torch
import numpy as np
import torch
import wandb
import time
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="main_config.yaml")
def main_loop(cfg):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    n_budget_step = 1_000_000
    print_interval = n_budget_step // 100

    env_class = task_map[cfg["env_name"]]
    grasp_env = env_class(
        cfg,
        sim_device="cuda" if cfg["sim"]["use_gpu_pipeline"] else "cpu",
        graphics_device_id=0,
        headless=cfg["headless"],
    )
    grasp_env.reset()
    hydra_cfg = HydraConfig.get()
    config = {
        "algo": "random",
        "env": cfg["env_name"],
        "scene": hydra_cfg.runtime.choices.scene,
        "command": hydra_cfg.runtime.choices.command,
        "num_env": grasp_env.num_envs,
        "ep_length": grasp_env.max_episode_length,
        "env_device": grasp_env.device,
        "rl_device": grasp_env.rl_device,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    wandb.init(
        project="Isaac-Manipulation", config=config, entity="oserris", mode="disabled"
    )

    creward = torch.zeros(grasp_env.num_envs, device=grasp_env.rl_device)
    n_step_done = 0
    start = time.time()
    while n_step_done < n_budget_step:
        n_step_done += grasp_env.num_envs
        actions = np.random.uniform(-1, 1, (grasp_env.num_envs, grasp_env.num_actions))
        actions = to_torch(actions, device=grasp_env.rl_device)
        observation, reward, done, info = grasp_env.step(actions)
        creward += reward
        if n_step_done % print_interval == 0:
            elapsed = time.time() - start
            step_per_seconds = n_step_done / elapsed
            print(
                f"n_steps : {n_step_done}/{n_budget_step} ({n_step_done/n_budget_step*100:.0f}%) \t sps : {step_per_seconds}",
            )
            wandb.log(
                {
                    "creward": creward.max(),
                    "step_per_seconds": step_per_seconds,
                },
                step=n_step_done,
            )
        creward[done] = 0


if __name__ == "__main__":
    main_loop()
