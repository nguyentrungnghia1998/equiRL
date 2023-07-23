import numpy as np
import torch
import os
import time
import json
import copy

from equi import utils
from equi.logger import Logger
import escnn
from equi.equi_agent import SacAgent, SACfD
from equi.equi_agent import ActorEquivariant_1
from equi.default_config import DEFAULT_CONFIG

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import wandb
import gc
from utils import RNN_actor, BC_RNN_actor, BC_RNN_GMM_actor
from torch.utils.data import TensorDataset, DataLoader 
import pandas as pd

def test_equi(message, obs, picker_state, action, agent):
    act = agent.actor.act
    if message == 'Test equivariant encoder':
        # encoder = PixelEncoderEquivariant(obs_shape=obs_shape, feature_dim=feature_dim, N=N, num_filters=num_filters)
        encoder = agent.actor.encoder
        field = escnn.nn.FieldType(act, obs.shape[-2]*[act.trivial_repr])
        y = encoder(escnn.nn.GeometricTensor(obs, field))
        print('='*50)
        print('Test equivariant encoder')
        print('='*50)

        x = field(obs)

        for i, g in enumerate(act.testing_elements):
            print(i, g)
            # transform y by g
            y_tr = y.transform(g)

            # transform x by g
            x_tr = x.transform(g).tensor
            y_ = encoder(escnn.nn.GeometricTensor(x_tr, field))

            print(y_.tensor.reshape(1, -1))
            print(y_tr.tensor.reshape(1, -1))
            assert torch.allclose(y_.tensor, y_tr.tensor, atol=1e-2)
            print('OK')
            print('-'*50)

    elif message == 'Test equivariant actor':
        print('='*50)
        print('Test equivariant actor')
        print('='*50)
        # actor = ActorEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, log_std_min=-20, log_std_max=2, num_layers=1, num_filters=16, N=N)
        actor = agent.actor
        out, _, _, ls = actor(obs)
        x = escnn.nn.FieldType(act, obs.shape[-2]*[act.trivial_repr])(obs)
        out = torch.cat([out[:, 0:1], out[:, 2:3], out[:, 4:5], out[:, 6:7], out[:, 1:2], out[:, 3:4], out[:, 5:6], out[:, 7:8], ls], dim=1).reshape(1, -1, 1, 1)
        out_type = escnn.nn.FieldType(act, 2*[act.irrep(1)] + 12*[act.trivial_repr])

        out_ = out_type(out)
        for i, g in enumerate(act.testing_elements):
            print(i, g)
            out_tr = out_.transform(g).tensor.reshape(1, -1)
            out_tr = torch.cat([out_tr[:, 0:1], out_tr[:, 4:5], out_tr[:, 1:2], out_tr[:, 5:6], out_tr[:, 2:3], out_tr[:, 6:7], out_tr[:, 3:4], out_tr[:, 7:8], out_tr[:, 8:]], dim=1)

            x_tr = x.transform(g)
            out_x_tr, _, _, ls_x_tr = actor(x_tr.tensor)
            out_x_tr_ = torch.cat([out_x_tr, ls_x_tr], dim=1)

            print(out_tr)
            print(out_x_tr_)
            assert torch.allclose(out_x_tr_, out_tr, atol=1e-2)
            print('OK')
            print('-'*50)

    elif message == 'Test equivariant actor 1':
        # import ipdb; ipdb.set_trace()
        print('='*50)
        print('Test equivariant actor 1')
        print('='*50)
        # actor = ActorEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, log_std_min=-20, log_std_max=2, num_layers=1, num_filters=16, N=N)
        actor = agent.actor
        out, _, _, ls = actor(obs, picker_state)
        x = escnn.nn.FieldType(act, obs.shape[-3]*[act.trivial_repr])(obs)
        picker_ = escnn.nn.FieldType(act, 2*[act.trivial_repr])(picker_state.view(1, 2, 1, 1))
        out = torch.cat([out[:, 0:1], out[:, 2:3], out[:, 4:5], out[:, 6:7], out[:, 1:2], out[:, 3:4], out[:, 5:6], out[:, 7:8], ls], dim=1).reshape(1, -1, 1, 1)
        out_type = escnn.nn.FieldType(act, 2*[act.irrep(1)] + 12*[act.trivial_repr])

        out_ = out_type(out)
        for i, g in enumerate(act.testing_elements):
            print(i, g)
            out_tr = out_.transform(g).tensor.reshape(1, -1)
            out_tr = torch.cat([out_tr[:, 0:1], out_tr[:, 4:5], out_tr[:, 1:2], out_tr[:, 5:6], out_tr[:, 2:3], out_tr[:, 6:7], out_tr[:, 3:4], out_tr[:, 7:8], out_tr[:, 8:]], dim=1)

            x_tr = x.transform(g)
            picker_tr = picker_.transform(g)
            out_x_tr, _, _, ls_x_tr = actor(x_tr.tensor, picker_tr.tensor)
            out_x_tr_ = torch.cat([out_x_tr, ls_x_tr], dim=1)

            # print(out_tr)
            # print(out_x_tr_)
            # assert torch.allclose(out_x_tr_, out_tr, atol=1e-2)
            print(torch.mean((out_x_tr_ - out_tr)**2))
            print('OK')
            print('-'*50)

    elif message == 'Test equivariant critic':
        print('='*50)
        print('Test equivariant critic')
        print('='*50)
        # action = torch.randn(1, 8)
        # critic = CriticEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, num_layers=1, num_filters=16, N=N)
        critic = agent.critic
        out1, out2 = critic(obs, action)
        x = escnn.nn.FieldType(act, obs.shape[-3]*[act.trivial_repr])(obs)
        action_ = torch.cat([action[:, 1:2], action[:, 3:4], action[:, 5:6], action[:, 7:8], action[:, 0:1], action[:, 2:3], action[:, 4:5], action[:, 6:7]], dim=1).reshape(1, 8, 1, 1)
        action = escnn.nn.FieldType(act, 4 * [act.trivial_repr] + 2*[act.irrep(1)])(action_)

        for i, g in enumerate(act.testing_elements):
            print(i, g)
            out1_tr, out2_tr = out1, out2
            x_tr = x.transform(g)
            a_tr = action.transform(g).tensor
            a_tr_ = torch.cat([a_tr[:, 4:5, :, :], a_tr[:, 0:1, :, :], a_tr[:, 5:6, :, :], a_tr[:, 1:2, :, :], a_tr[:, 6:7, :, :], a_tr[:, 2:3, :, :], a_tr[:, 7:8, :, :], a_tr[:, 3:4, :, :]], dim=1).reshape(1, 8)
            out1_, out2_ = critic(x_tr.tensor, a_tr_)
            print(out1_)
            print(out1_tr)
            # print mse
            # print(torch.mean((out1_ - out1_tr)**2))
            assert torch.allclose(out1_, out1_tr, atol=1e-1)
            assert torch.allclose(out2_, out2_tr, atol=1e-1)
            print('OK')
            print('-'*50)

    elif message == 'Test equivariant critic 1':
        # import ipdb; ipdb.set_trace()
        print('='*50)
        print('Test equivariant critic 1')
        print('='*50)
        # action = torch.randn(1, 8)
        # critic = CriticEquivariant(obs_shape=obs_shape, action_shape=(8,), hidden_dim=feature_dim, encoder_type='pixel-equivariant', encoder_feature_dim=feature_dim, num_layers=1, num_filters=16, N=N)
        critic = agent.critic
        out1, out2 = critic(obs, picker_state, action)
        x = escnn.nn.FieldType(act, obs.shape[-3]*[act.trivial_repr])(obs)
        picker_ = escnn.nn.FieldType(act, 2*[act.trivial_repr])(picker_state.view(1, 2, 1, 1))
        action_ = torch.cat([action[:, 1:2], action[:, 3:4], action[:, 5:6], action[:, 7:8], action[:, 0:1], action[:, 2:3], action[:, 4:5], action[:, 6:7]], dim=1).reshape(1, 8, 1, 1)
        action = escnn.nn.FieldType(act, 4 * [act.trivial_repr] + 2*[act.irrep(1)])(action_)

        for i, g in enumerate(act.testing_elements):
            print(i, g)
            out1_tr, out2_tr = out1, out2
            x_tr = x.transform(g)
            picker_tr = picker_.transform(g)
            a_tr = action.transform(g).tensor
            a_tr_ = torch.cat([a_tr[:, 4:5, :, :], a_tr[:, 0:1, :, :], a_tr[:, 5:6, :, :], a_tr[:, 1:2, :, :], a_tr[:, 6:7, :, :], a_tr[:, 2:3, :, :], a_tr[:, 7:8, :, :], a_tr[:, 3:4, :, :]], dim=1).reshape(1, 8)
            out1_, out2_ = critic(x_tr.tensor, picker_tr.tensor, a_tr_)
            # print(out1_)
            # print(out1_tr)
            print(torch.mean((out1_ - out1_tr)**2))
            # assert torch.allclose(out1_, out1_tr, atol=1e-1)
            # assert torch.allclose(out2_, out2_tr, atol=1e-1)
            print('OK')
            print('-'*50)

    else:
        print('Wrong message')

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
            count = 0
            while not done:
                if args.encoder_type == 'pixel':
                    if obs.shape[0] == 1:
                        obs = obs[0]
                    
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs, picker_state)
                    else:
                        action = agent.select_action(obs, picker_state)
                obs, reward, done, info = env.step(action)
                picker_state = utils.get_picker_state(env)
                episode_reward += reward
                count += 1
                ep_info.append(info)
                frames.append(env.get_image(128, 128))
                rewards.append(reward)
                if done:
                    for i in range(env.horizon + 1 - count):
                        frames.append(env.get_image(128, 128))
                if count == env.horizon:
                    done = True
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
    if args.agent == 'sac':
        
        return SacAgent(
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
            num_rotations=args.num_rotations
        )
    elif args.agent == 'sacfd':
        return SACfD(
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
            num_rotations=args.num_rotations
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def test_phase(env, actor, args, device, epoch, average_rewards, all_epoch, hidden_size = 128):
    actor.eval()
    test_episode = 8
    sample_stochastically = True
    infos = []
    all_frames = []
    final_rewards = []
    plt.figure()
    with torch.no_grad():
        for i in range(test_episode):
            obs = env.reset(eval_flag = True)
            obs = obs/255.0
            obs = obs.to(device)
            picker_state = utils.get_picker_state(env)
            done = False
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(128, 128)]
            rewards = []
            count = 0
            h0 = torch.zeros(2,1,hidden_size).to(device)
            c0 = torch.zeros(2,1,hidden_size).to(device)
            while not done:
                # if args.encoder_type == 'pixel':
                #     if obs.shape[0] == 1:
                #         obs = obs[0]
                    
                # with utils.eval_mode(agent):
                #     if sample_stochastically:
                #         action = agent.sample_action(obs, picker_state)
                #     else:
                #         action = agent.select_action(obs, picker_state)
                obs = obs.unsqueeze(dim = 1)
                picker_state = torch.Tensor(picker_state).unsqueeze(dim = 0).unsqueeze(dim = 0)
                picker_state = picker_state.repeat_interleave(50,dim = 2)
                picker_state = picker_state.to(device)
                if args.use_GMM:
                    dist, hn, cn = actor(obs, picker_state, (h0, c0), train = False)
                    action = dist.sample()
                    action = action.cpu().numpy()
                else:
                    action, hn, cn = actor(obs, picker_state, (h0,c0))
                    action = action.detach().cpu().numpy()
                obs, reward, done, info = env.step(action)
                obs = obs/255.0
                obs = obs.to(device)
                picker_state = utils.get_picker_state(env)
                episode_reward += reward
                count += 1
                ep_info.append(info)
                frames.append(env.get_image(128, 128))
                rewards.append(info['normalized_performance'])
                if done:
                    for i in range(env.horizon + 1 - count):
                        frames.append(env.get_image(128, 128))
                if count == env.horizon:
                    done = True
                h0 = hn
                c0 = cn
            final_rewards.append(rewards[-1])
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)
            infos.append(ep_info)

    average_rewards.append(np.mean(final_rewards))
    all_epoch.append(epoch)
    
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    plt.savefig(f"data/RNN_imitation/results/reward_epoch_{epoch}.png")

    all_frames = np.array(all_frames).swapaxes(0, 1)
    all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
    save_numpy_as_gif(all_frames, f"data/RNN_imitation/results/policy_epoch_{epoch}.gif")

def main(args):
    torch.cuda.empty_cache()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs
    
    if args.wandb:
        # ed44c646a708f75a7fe4e39aee3844f8bfe44858
        wandb.login(key='ed44c646a708f75a7fe4e39aee3844f8bfe44858')
        group_name = args.exp_name + '_aug' if args.aug_transition else args.exp_name + '_no_aug'
        wandb.init(project=args.env_name, settings=wandb.Settings(_disable_stats=True), group=group_name, name=f's{args.wandb_seed}', entity='longdinh')
    else:
        print('==================== NOT USING WANDB ====================')

    symbolic = False if args.env_kwargs['observation_mode'] in ['cam_rgb', 'img_depth', 'only_depth'] else True
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
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
            n += args.env_kwargs['num_picker']
        obs_shape = (n, args.image_size, args.image_size)
        pre_aug_obs_shape = (n, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    if args.aug_transition:
        if args.prioritized_replay:
            print(f'================ AUGMENTED TRANSITION with {args.aug_n} TRANSFORMATION of {args.aug_type} and PRIORITIZED REPLAY =================')
            replay_buffer = utils.PrioritizedReplayBufferAugmented(
                obs_shape=pre_aug_obs_shape,
                action_shape=action_shape,
                capacity=args.replay_buffer_capacity,
                batch_size=args.batch_size,
                device=device,
                image_size=args.image_size,
                aug_n = args.aug_n,
                alpha=args.prioritized_replay_alpha,
            )
            p_beta_schedule = utils.LinearSchedule(schedule_timesteps=args.num_train_steps, initial_p=args.per_beta, final_p=1.0)

        else:
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
    if args.use_GMM:
        actor = BC_RNN_GMM_actor(obs_shape = (1,128,128), device = device).to(device)
    else:
        actor = BC_RNN_actor(obs_shape=(1,128,128), device=device).to(device)
    if args.collect_demonstration:
        utils.create_demonstration(env, video_dir=video_dir, num_demonstrations=args.num_demonstrations)
    obses = []
    actions = []
    rewards = []
    picker_states = []
    df_demon = pd.read_csv("data/RNN_imitation/video/demo.csv")
    for path in df_demon['NPY_Path']:
        r = []
        obs_i = []
        action_i = []
        picker_state_i = []
        dataset = np.load(path, allow_pickle=True)
        if dataset.shape[0]>=args.train_length:
            for j in range(1,args.train_length + 1):
                data = dataset[-j,:]
                r.insert(0,data[2])
                obs_i.insert(0,data[0])
                action_i.insert(0,data[1])
                picker_state_i.append(data[5])
        else:
            for j in range(dataset.shape[0]):
                data = dataset[j,:]
                r.append(data[2])
                obs_i.append(data[0])
                action_i.append(data[1])
                picker_state_i.append(data[5])
            data = dataset[0,:]
            for j in range(args.train_length - dataset.shape[0]):
                r.insert(0,data[2])
                obs_i.insert(0,data[0])
                action_i.insert(0,data[1])
                picker_state_i.append(data[5])
        plt.plot(range(len(r)), r)  
        obses.append(obs_i)
        actions.append(action_i)
        rewards.append(r)
        picker_states.append(picker_state_i)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    plt.savefig("data/RNN_imitation/video/reward.png")
    plt.show()  

    plt.close()  


    obses = torch.stack([torch.cat(obs) for obs in obses], dim = 0)
    obses = obses/255.0
    picker_states = torch.Tensor(picker_states)
    picker_states = picker_states.repeat_interleave(50, dim = 2)
    actions = torch.FloatTensor(actions)

    dataset = TensorDataset(obses, picker_states, actions)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(actor.parameters(),lr = args.actor_lr, betas= (args.actor_beta,0.999))
    steps = []
    losses = []
    # for i in range(20000):
    #     print(f'Train {i} step')
    #     obs, picker_state, action, reward, next_obs, next_picker_state, not_done = replay_buffer.sample_proprio()

    #     if i % args.log_interval == 0:
    #         L.log('train/batch_reward', reward.mean(), i)
    #         if args.wandb:
    #             wandb.log({'train_batch_reward': reward.mean()}, step=i)
        
    #     if i % args.actor_update_freq == 0: #default actor_update_freq = 2
    #         # self.update_actor_and_alpha(obs, L, step)
    #         _, pi, log_pi, log_std = agent.actor(obs, picker_state)
    #         actor_loss = criterion(pi,action.detach())

    #         if i % args.log_interval == 0:
    #             L.log('train_actor/loss', actor_loss, i)
    #             if args.wandb:
    #                 wandb.log({'train_actor_loss': actor_loss}, step=i)
    #         print("Actor loss: ", actor_loss.item())
    #         losses.append(actor_loss.item())
    #         steps.append(i)
    #         optimizer.zero_grad()
    #         actor_loss.backward()
    #         optimizer.step()
    # plt.plot(steps,losses)
    # plt.savefig("data/imitation/actor_loss.png")
    # plt.show()

    average_rewards = []
    all_epoch = []
    if not os.path.exists("data/RNN_imitation/results/"):
        os.mkdir("data/RNN_imitation/results/")

    for epoch in range(args.num_train_steps):
        print(f'Train {epoch} step')
        actor.train()
        running_loss = 0.0
        steps.append(epoch)
        for i, (obs, picker_state, action) in enumerate(data_loader):
            obs, picker_state, action = obs.to(device), picker_state.to(device), action.to(device)
            optimizer.zero_grad()
            outputs, _, _ = actor(obs, picker_state)
            if args.use_GMM:
                log_probs = outputs.log_prob(action)
                loss = -log_probs.mean()
            else:
                loss = criterion(outputs,action)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss/len(data_loader))
        print(f"Epoch {epoch} train loss: {running_loss/len(data_loader)}")
        if epoch % args.every_test == 0:
            test_phase(env, actor, args, device, epoch, average_rewards, all_epoch)

    # Create a new figure, plot into it, then close it so it never gets displayed
    plt.figure()
    plt.plot(steps,losses)
    plt.savefig("data/RNN_imitation/actor_loss.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(all_epoch, average_rewards)
    plt.savefig("data/RNN_imitation/average_rewards.png")
    plt.show()
    plt.close()



    torch.save(actor.state_dict(),"data/RNN_imitation/actor_final.pt")