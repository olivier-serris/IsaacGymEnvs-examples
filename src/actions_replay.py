import isaacgym  # importing isaac_gym before torch is mandatory.
import os
from isaac_gym_manipulation.envs import task_map
import hydra
from omegaconf import OmegaConf
from isaacgym.torch_utils import to_torch
import numpy as np
import torch
import time
from hydra.core.hydra_config import HydraConfig
import wandb


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="main_config.yaml")
def main_loop(cfg):
    api = wandb.Api()
    run = api.run("oserris/Isaac-Manipulation/3qcx4py1")  # bad grasp : 3qcx4py1
    run.file("best_actions").download(replace=True)
    best_actions = torch.load("best_actions")

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["env"]["numEnvs"] = 1

    env_class = task_map[cfg["env_name"]]
    grasp_env = env_class(
        cfg,
        sim_device="cpu",  # "cuda" if cfg["sim"]["use_gpu_pipeline"] else "cpu",
        graphics_device_id=0,
        headless=False,
    )
    grasp_env.reset(with_noise=False)
    while True:
        creward = torch.zeros(grasp_env.num_envs, device=grasp_env.rl_device)
        for _ in range(grasp_env.max_episode_length):

            actions = (
                best_actions[grasp_env.progress_buf[0]]
                .repeat((grasp_env.num_envs, 1))
                .clone()
                .to(grasp_env.rl_device)
            )
            actions = to_torch(actions, device=grasp_env.rl_device)
            observation, reward, done, info = grasp_env.step(actions)
            creward += reward

        top10 = torch.topk(creward, min(grasp_env.num_envs, 10))
        print(
            f"best reward :{creward.max().item()} \t",
            f"mean over all envs {creward.mean().item():.1f} \t",
            f"topk env rewards { {k.item():v.item() for k,v in zip(top10.indices,top10.values)} }",
        )
        grasp_env.reset(with_noise=False)


if __name__ == "__main__":
    main_loop()
