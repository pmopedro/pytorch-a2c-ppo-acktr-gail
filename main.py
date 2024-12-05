import copy
import glob
import os
import time
from collections import deque
import yaml
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_test
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime  # For timestamp
import subprocess  # For Git commit hash


def main():
    args = get_args()

    # Print all arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Convert arguments to a dictionary
    args_dict = vars(args)

    # Add metadata
    args_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args_dict['git_commit'] = git_commit
    except Exception:
        args_dict['git_commit'] = 'N/A'

    # Set up logging directories
    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # Write arguments to a YAML metadata file
    metadata_file = os.path.join(log_dir, 'metadata.yml')
    os.makedirs(log_dir, exist_ok=True)
    with open(metadata_file, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    # Log configuration to TensorBoard
    config_text = yaml.dump(args_dict, default_flow_style=False)
    writer.add_text('Configuration', f"```\n{config_text}\n```")

    # Set seeds and device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Initialize kl_loss_coef
    kl_loss_coef = args.kl_loss_coef_init

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        print("Using A2C with calculation")
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            kl_loss_coef=args.kl_loss_coef_init)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    pre_train_finished = False
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if not pre_train_finished and total_num_steps >= args.pretrain_steps:
            # Finish pretraining
            pre_train_finished = True
            print("================== Finished Pretraining ==================")

        if pre_train_finished and kl_loss_coef <= 1:
            # Update kl_loss_coef using annealing scheme
            kl_loss_coef *= args.kl_loss_coef_alpha

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, kl_divergence = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            # # Convert reward to tensor and ensure it has shape [num_processes, 1]
            # if isinstance(reward, np.ndarray):
            #     reward = torch.from_numpy(reward).float().to(device)
            # else:
            #     reward = reward.float().to(device)

            # reward = reward.view(-1, 1)

            # # Ensure kl_divergence has shape [num_processes, 1]
            # kl_divergence = kl_divergence.view(-1, 1)

            # # Modify the reward by adding KL divergence
            # reward = reward + kl_divergence

            # Proceed as before
            masks = torch.FloatTensor(
                list([[0.0] if done_ else [1.0] for done_ in done])).to(device)
            bad_masks = torch.FloatTensor(
                list([[0.0] if 'bad_transition' in info.keys() else [1.0]
                      for info in infos])).to(device)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # Insert the data into rollouts
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, kl_loss, total_norm = agent.update(
            rollouts, kl_loss_coef)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n"
                "Value loss: {:.5f}, Action loss: {:.5f}, Entropy: {:.5f}, KL Loss: {:.5f}, KL Coef: {:.5f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards),
                        value_loss, action_loss, dist_entropy, kl_loss, kl_loss_coef))

            writer.add_scalar('Loss/Value Loss', value_loss, total_num_steps)
            writer.add_scalar('Loss/Action Loss', action_loss, total_num_steps)
            writer.add_scalar('Loss/Entropy', dist_entropy, total_num_steps)
            writer.add_scalar('Reward/Mean Episode Reward',
                              np.mean(episode_rewards), total_num_steps)
            writer.add_scalar('Reward/Median Episode Reward',
                              np.median(episode_rewards), total_num_steps)
            writer.add_scalar('Reward/Min Episode Reward',
                              np.min(episode_rewards), total_num_steps)
            writer.add_scalar('Reward/Max Episode Reward',
                              np.max(episode_rewards), total_num_steps)
            writer.add_scalar('Gradient Norm', total_norm, total_num_steps)
            # Log KL Coefficient
            writer.add_scalar('Loss/KL Coefficient',
                              kl_loss_coef, total_num_steps)

            # Compute the mean kl
            mean_kl_divergence = kl_divergence.mean()
            writer.add_scalar(
                'KL_Divergence', mean_kl_divergence, total_num_steps)
            writer.add_scalar('Loss/KL Loss', kl_loss, total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    test_envs = make_vec_envs_test(args.env_name, args.seed, args.num_processes,
                                   args.gamma, args.log_dir, device, False)
    pre_train_finished = False

    print("Evaluating")
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = test_envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    for j in range(num_updates, 2*num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if pre_train_finished and kl_loss_coef <= 1:
            # Update kl_loss_coef using annealing scheme
            kl_loss_coef *= args.kl_loss_coef_alpha

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, kl_divergence = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = test_envs.step(action)

            # # Convert reward to tensor and ensure it has shape [num_processes, 1]
            # if isinstance(reward, np.ndarray):
            #     reward = torch.from_numpy(reward).float().to(device)
            # else:
            #     reward = reward.float().to(device)

            # reward = reward.view(-1, 1)

            # # Ensure kl_divergence has shape [num_processes, 1]
            # kl_divergence = kl_divergence.view(-1, 1)

            # # Modify the reward by adding KL divergence
            # reward = reward + kl_divergence

            # Proceed as before
            masks = torch.FloatTensor(
                list([[0.0] if done_ else [1.0] for done_ in done])).to(device)
            bad_masks = torch.FloatTensor(
                list([[0.0] if 'bad_transition' in info.keys() else [1.0]
                      for info in infos])).to(device)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # Insert the data into rollouts
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                test_envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(test_envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, kl_loss, total_norm = agent.update(
            rollouts, kl_loss_coef)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(test_envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n"
                "Value loss: {:.5f}, Action loss: {:.5f}, Entropy: {:.5f}, KL Loss: {:.5f}, KL Coef: {:.5f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards),
                        value_loss, action_loss, dist_entropy, kl_loss, kl_loss_coef))

            writer.add_scalar('Loss/Value Loss', value_loss, total_num_steps)
            writer.add_scalar('Loss/Action Loss', action_loss, total_num_steps)
            writer.add_scalar('Loss/Entropy', dist_entropy, total_num_steps)
            writer.add_scalar('Reward/Mean Episode Reward',
                              np.mean(episode_rewards), total_num_steps)
            writer.add_scalar('Reward/Median Episode Reward',
                              np.median(episode_rewards), total_num_steps)
            writer.add_scalar('Reward/Min Episode Reward',
                              np.min(episode_rewards), total_num_steps)
            writer.add_scalar('Reward/Max Episode Reward',
                              np.max(episode_rewards), total_num_steps)
            writer.add_scalar('Gradient Norm', total_norm, total_num_steps)
            # Log KL Coefficient
            writer.add_scalar('Loss/KL Coefficient',
                              kl_loss_coef, total_num_steps)

            # Compute the mean kl
            mean_kl_divergence = kl_divergence.mean()
            writer.add_scalar(
                'KL_Divergence', mean_kl_divergence, total_num_steps)
            writer.add_scalar('Loss/KL Loss', kl_loss, total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(test_envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
