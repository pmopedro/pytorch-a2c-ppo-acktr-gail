from gymnasium import ObservationWrapper, spaces
from gymnasium import spaces
import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
from gymnasium.wrappers.clip_action import ClipAction
from a2c_ppo_acktr.minigrid_env import SimpleEnv
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

# try:
#     import pybullet_envs
# except ImportError:
#     pass

# from gymnasium.envs.registration import register


import os
import pickle
import matplotlib.pyplot as plt


def make_env(env_id, rank, log_dir, allow_early_resets, seed):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dmc2gym.make(domain_name=domain, task_name=task)
            env = ClipAction(env)
        else:
            env = SimpleEnv(size=13, replace_agent_episode=True,
                            replace_goal_episode=False, seed=seed)
            env = ImageObservationWrapper(env)
            env = TransposeImage(env)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            # Create directory for envs if it doesn't exist
            envs_dir = os.path.join(log_dir, 'envs')
            os.makedirs(envs_dir, exist_ok=True)

            # Save the environment instance as a pickle
            env_pickle_path = os.path.join(envs_dir, f"env_rank_{rank}.pkl")
            with open(env_pickle_path, 'wb') as f:
                pickle.dump(env, f)

            # Save the environment visualization as a PNG
            env_png_path = os.path.join(envs_dir, f"env_rank_{rank}.png")
            save_env_as_png(env, env_png_path)

            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = EpisodicLifeEnv(env)
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env, width=84, height=84)
                env = ClipRewardEnv(env)
        else:
            print(
                f"Custom env created with observation_space: {env.observation_space}")

        return env

    return _thunk


def save_env_as_png(env, file_path):
    """
    Save a rendered visualization of the environment as a PNG file.
    """
    grid = env.gen_obs()["image"]  # Get the grid visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(grid)  # Render the grid as an image
    plt.axis("off")
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, i, log_dir, allow_early_resets, seed=i)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, norm_reward=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    else:
        pass
        # envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info, _ = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info, _

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(ObservationWrapper):
    def __init__(self, env):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        # Transpose the shape to (channels, height, width)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        # Transpose the actual observation
        return observation.transpose(2, 0, 1)


class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageObservationWrapper, self).__init__(env)
        # Update the observation space to be just the 'image' space
        self.observation_space = env.observation_space['image']

    def observation(self, observation):
        # Return only the 'image' part of the observation
        return observation['image']


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/masfter/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
