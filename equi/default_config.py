DEFAULT_CONFIG = {
    # environment
    'domain_name': 'cartpole',
    'task_name': 'swingup',
    'pre_transform_image_size': 224,
    'image_size': 224,
    'action_repeat': 1,
    # replay buffer
    'replay_buffer_capacity': 100000,
    # train
    'agent': 'sacfd',
    'init_steps': 0,
    'num_train_steps': 2000000,
    'batch_size': 128,  # This is 256 for other sac but experiments show that they don't matter much for curl_sac
    'hidden_dim': 128,
    # eval
    'eval_freq': 5000,
    'num_eval_episodes': 10,
    # critic
    'critic_lr': 1e-3,
    'critic_beta': 0.9,
    'critic_tau': 0.01,  # try 0.05 or 0.1
    'critic_target_update_freq': 2,  # try to change it to 1 and retain 0.01 above
    # actor
    'actor_lr': 1e-3,
    'actor_beta': 0.9,
    'actor_log_std_min': -10,
    'actor_log_std_max': 2,
    'actor_update_freq': 2,
    # encoder
    'encoder_type': 'pixel',
    'encoder_feature_dim': 128,
    'encoder_lr': 1e-3,
    'encoder_tau': 0.05,
    'num_layers': 4,
    'num_filters': 16,
    'curl_latent_dim': 128,
    # sac
    'discount': 0.99,
    'init_temperature': 0.1,
    'alpha_lr': 1e-4,
    'alpha_beta': 0.5,
    'alpha_fixed': False,
    # misc
    'seed': 1,
    'save_tb': False,
    'save_buffer': False,
    'save_video': False,
    'save_model': False,
    'detach_encoder': False,
    'log_interval': 1,
    # augment transition
    'aug_transition': True,
    'aug_n': 4,
    'aug_type': 'so2',
    # equivariant
    'num_rotations': 4,
    # wandb
    'wandb': False,
    'wandb_seed': 15,
    # expert
    'num_demonstrations': 40,
    # prioritized replay
    'prioritized_replay': False,
    'prioritized_replay_alpha': 0.6,
    'per_beta': 0.4,
    'per_eps': 1e-6,
    'per_expert_eps': 1.0,
    # pretrain
    'pretrain': True,
    'pretrain_steps': 100000,
}
