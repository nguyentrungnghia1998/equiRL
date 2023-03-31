import numpy as np
import torch
import os
import time
import json
import copy

from curl import utils
from curl.logger import Logger

from curl.curl_sac import CurlSacAgent
from curl.default_config import DEFAULT_CONFIG

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

import wandb

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters
    # with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
    #     json.dump(vv, f, indent=2, sort_keys=True)

    return args


def run_task(vv, log_dir=None, exp_name=None):
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    args = vv_to_args(updated_vv)
    if args.wandb:
        log_dir = os.path.join(log_dir, f's{args.wandb_seed}')
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    os.makedirs(logdir, exist_ok=True)
    assert logdir is not None
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(updated_vv, f, indent=2, sort_keys=True)
    main(args)

def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict


def evaluate(env, agent, video_dir, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        infos = []
        all_frames = []
        plt.figure()
        for i in range(num_episodes):
            obs = env.reset(eval_flag=True)
            picker_state = utils.get_picker_state(env)
            done = False
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(128, 128)]
            rewards = []
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs,picker_state)
                    else:
                        action = agent.select_action(obs,picker_state)
                # action = np.array([1.0, 0.0, 0.0, 0.15, -1.0, 0.0, 0.0, 0.01])
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                ep_info.append(info)
                frames.append(env.get_image(128, 128))
                rewards.append(reward)
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)
            infos.append(ep_info)

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title('Reward over time')
        plt.savefig(os.path.join(video_dir, '%d.png' % step))
        plt.close()        
        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
        save_numpy_as_gif(all_frames, os.path.join(video_dir, '%d.gif' % step))

        for key, val in get_info_stats(infos).items():
            L.log('eval/info_' + prefix + key, val, step)
            if args.wandb:
                wandb.log({key:val},step = step)

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    

def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            alpha_fixed=args.alpha_fixed,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main(args):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    if args.wandb:
        # ed44c646a708f75a7fe4e39aee3844f8bfe44858
        group_name = args.exp_name + '_aug' if args.aug_transition else args.exp_name + '_no_aug'
        wandb.init(project=args.env_name, settings=wandb.Settings(_disable_stats=True), group=group_name, name=f's{args.wandb_seed}', entity='longdinh')
    else:
        print('Not using wandb')

    symbolic = args.env_kwargs['observation_mode'] not in ['cam_rgb','img_depth','only_depth']
    args.encoder_type = 'identity' if symbolic else 'pixel'
    # import ipdb; ipdb.set_trace()
    env = Env(args.env_name, symbolic, args.seed, 100, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)

    env.seed(args.seed)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        if args.env_kwargs['observation_mode'] == 'cam_rgb':
            n = 3
        elif args.env_kwargs['observation_mode'] == 'only_depth':
            n = 1
        else:
            n = 4

        if args.env_kwargs['use_picker_state']:
            n+=args.env_kwargs['num_picker']
        obs_shape = (n, args.image_size, args.image_size)
        pre_aug_obs_shape = (n, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    if args.aug_transition:
        print(f'==================== AUGMENTED TRANSITION with {args.aug_n} TRANSFORMATION of {args.aug_type}====================')
        replay_buffer = utils.ReplayBufferAugmented(
            obs_shape=pre_aug_obs_shape,
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
            aug_n = args.aug_n,
        )
    else:
        print('================ DON NOT USE AUGMENTED TRANSITION =================')
        replay_buffer = utils.ReplayBuffer(
            obs_shape=pre_aug_obs_shape,
            action_shape=action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            image_size=args.image_size,
        )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    print('==================== START COLLECTING DEMONSTRATIONS ====================')
    all_frames_planner = []
    thresh = env.cloth_particle_radius + env.action_tool.picker_radius + env.action_tool.picker_threshold
    count_planner = 0
    while True:
        obs = env.reset()
        picker_state = utils.get_picker_state(env)
        episode_step = 0
        frames = [env.get_image(128, 128)]
        while True:
            # choose random boundary point
            choosen_id = utils.choose_random_particle_from_boundary(env)
            if choosen_id is None:
                print('[INFO] Cannot find boundary point!!!')
                break
            # move to two choosen boundary points and pick them
            pick_choosen = utils.pick_choosen_point(env, obs, picker_state, choosen_id, thresh, episode_step, frames, replay_buffer)
            if pick_choosen is None:
                count_planner += 1
                break
            if pick_choosen == 1:
                # release the cloth
                release = utils.give_up_the_cloth(env, obs, picker_state, episode_step, frames, replay_buffer)
                if release is None:
                    count_planner += 1
                    break
                episode_step, obs = release[0], release[1]
                continue
            else:
                episode_step, obs = pick_choosen[0], pick_choosen[1]
            # choose fling primitive or pick&drag primitive
            if np.random.rand() < 0.5:
                # fling primitive
                fling = utils.fling_primitive(env, obs, picker_state, choosen_id, thresh, episode_step, frames, replay_buffer)
                if fling is None:
                    count_planner += 1
                    break
                episode_step, obs = fling[0], fling[1]
            else:
                # pick&drag primitive
                pick_drag = utils.pick_drag_primitive(env, obs, picker_state, choosen_id, thresh, episode_step, frames, replay_buffer)
                if pick_drag is None:
                    count_planner += 1
                    break
                episode_step, obs = pick_drag[0], pick_drag[1]
            # release the cloth
            release = utils.give_up_the_cloth(env, obs, picker_state, episode_step, frames, replay_buffer)
            if release is None:
                count_planner += 1
                break
            episode_step, obs = release[0], release[1]
        all_frames_planner.append(frames)
        print('[INFO]Collected {} demonstrations'.format(count_planner))
        if count_planner == 20:
            print('==================== FINISH COLLECTING DEMONSTRATIONS ====================')
            break

    # final_reward.append(reward)
    all_frames_planner = np.array(all_frames_planner).swapaxes(0, 1)
    all_frames_planner = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames_planner])
    save_numpy_as_gif(all_frames_planner, os.path.join(video_dir, 'expert.gif'))

    episode, episode_reward, done, ep_info = 0, 0, True, []
    start_time = time.time()
    for step in range(args.num_train_steps):
        # evaluate agent periodically
        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video_dir, args.num_eval_episodes, L, step, args)
            if args.save_model and (step % (args.eval_freq * 5) == 0):
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)
        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    for key, val in get_info_stats([ep_info]).items():
                        L.log('train/info_' + key, val, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            picker_state = utils.get_picker_state(env)
            done = False
            ep_info = []
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            agent.update(replay_buffer, L, step)
        next_obs, reward, done, info = env.step(action)

        next_picker_states = utils.get_picker_state(env)

        # allow infinit bootstrap
        ep_info.append(info)
        done_bool = 0 if episode_step + 1 == env.horizon else float(done)
        episode_reward += reward
        replay_buffer.add(obs, picker_state, action, reward, next_obs, next_picker_states, done_bool)

        obs = next_obs
        picker_state = next_picker_states
        episode_step += 1
