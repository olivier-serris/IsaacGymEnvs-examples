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
from utils import Orn_Uhlen
import json


def update_episode_action_tensor(
    episode_actions: torch.Tensor,
    new_actions: torch.Tensor,
    time_position: torch.Tensor,
):
    """
    all_action has shape    :   MaxStep x NumEnv x NumAction
    new_actions has shape   :   NumEnv x NumAction
    time_position has shape :   NumEnv , it contains the time position for each action to be added.
    (interested by simpler code doing the same thing)
    """
    num_envs, num_actions = episode_actions.shape[1:]
    index = time_position.cpu().detach().expand(1, num_actions, num_envs).moveaxis(1, 2)
    src = (
        new_actions.cpu()
        .detach()
        .expand(
            1,
            num_envs,
            num_actions,
        )
    )
    return episode_actions.scatter(dim=0, index=index, src=src)


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="main_config.yaml")
def main_loop(cfg):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    n_budget_step = cfg["train"]["n_budget_step"]

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
        "step_budget": n_budget_step,
    }
    wandb.init(
        project="Isaac-Manipulation",
        config=config,
        entity="oserris",
        mode=cfg["train"]["wandb_mode"],
    )
    config_save_path = os.path.join(wandb.run.dir, "config")
    with open(config_save_path, mode="w") as f:
        json.dump(cfg, f)
    wandb.save(f"config")

    start = time.time()
    creward = torch.zeros(grasp_env.num_envs, device=grasp_env.rl_device)
    n_step_done = 0
    best_score = float("-inf")
    episode_actions = torch.zeros(
        (
            grasp_env.max_episode_length,
            grasp_env.num_envs,
            grasp_env.num_actions,
        ),
        device="cpu",
    )
    best_actions = torch.zeros(
        (grasp_env.max_episode_length, grasp_env.num_actions), device="cpu"
    )
    noise_gen = Orn_Uhlen(grasp_env.num_envs * grasp_env.num_actions)
    while n_step_done < n_budget_step:
        n_step_done += grasp_env.num_envs
        #
        if cfg["train"]["agent"] == "uniform":
            actions = np.random.uniform(
                -1, 1, (grasp_env.num_envs, grasp_env.num_actions)
            )
        elif cfg["train"]["agent"] == "orn_uhlen":
            actions = (noise_gen.sample() - noise_gen.sample()).reshape(
                grasp_env.num_envs, grasp_env.num_actions
            )
        actions = to_torch(actions, device=grasp_env.rl_device)
        # if episode_actions[-1, ...].max()> 0:
        #     actions = episode_actions[grasp_env.progress_buf[0], ...]
        episode_actions = update_episode_action_tensor(
            episode_actions, actions, grasp_env.progress_buf
        )
        observation, reward, done, info = grasp_env.step(actions)
        creward += reward
        if done.any():
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
            max_traj_id = torch.argmax(creward[done])
            if creward[done][max_traj_id] > best_score:
                best_score = creward[done][max_traj_id]
                best_actions[:] = episode_actions[:, done, :][:, max_traj_id, :]
                path = os.path.join(wandb.run.dir, "best_actions")
                print("new best trajectory", creward[done][max_traj_id].item())
                print("saves to: \n", path)
                torch.save(best_actions, path)
                wandb.save("best_action")
                wandb.log({"best_score": best_score}, step=n_step_done)

        creward[done] = 0


if __name__ == "__main__":
    main_loop()
