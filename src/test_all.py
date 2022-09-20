import isaacgym  # importing isaac_gym before torch is mandatory.
import itertools
from hydra import compose
from omegaconf import OmegaConf
from isaac_gym_manipulation.envs import task_map
from omegaconf import OmegaConf
from isaacgym.torch_utils import to_torch
import hydra
import os
import multiprocessing
from utils import Orn_Uhlen
import torch
import numpy as np


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def main_loop(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    env_class = task_map[cfg["env_name"]]
    grasp_env = env_class(
        cfg,
        sim_device="cuda" if cfg["sim"]["use_gpu_pipeline"] else "cpu",
        graphics_device_id=0,
        headless=cfg["headless"],
    )
    grasp_env.reset()

    noise_gen = Orn_Uhlen(grasp_env.num_envs * grasp_env.num_actions)
    while True:
        actions = np.random.uniform(-1, 1, (grasp_env.num_envs, grasp_env.num_actions))
        # actions = noise_gen.sample().reshape(
        #     (grasp_env.num_envs, grasp_env.num_actions)
        # )
        # actions = torch.ones((grasp_env.num_envs, grasp_env.num_actions))
        actions = to_torch(actions, device=grasp_env.rl_device)
        observation, reward, done, info = grasp_env.step(actions)


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="main_config.yaml")
def main(c):
    multiprocessing.set_start_method("spawn")
    config_groups = [
        *product_dict(command=["franka_dof", "franka_effort"], scene=["franka_ycb"]),
        # *product_dict(command=["tiago_dof", "tiago_effort"], scene=["tiago_ycb"]),
    ]

    # impossible to create more than 1 isaac env.
    # so we can't use a for loop to test all params
    # for params in config_groups:
    params = config_groups[1]
    overrides = [f"{key}={value}" for key, value in params.items()]
    overrides.append("headless=False")
    cfg = compose(config_name="main_config", overrides=overrides)
    cfg["env"]["numEnvs"] = 8
    print("Test with : ", overrides)
    main_loop(cfg)


if __name__ == "__main__":
    main()
