import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import gym
import os
import random
import copy
from collections import deque
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
from collections import deque
from scipy.ndimage import affine_transform
from equi.default_config import DEFAULT_CONFIG
from equi.segment_tree import SumSegmentTree, MinSegmentTree
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import ConvexHull
from softgym.utils.visualization import save_numpy_as_gif, make_grid
import cv2
import pandas as pd



class ConvergenceChecker(object):
    def __init__(self, threshold, history_len):
        self.threshold = threshold
        self.history_len = history_len
        self.queue = deque(maxlen=history_len)
        self.converged = None

    def clear(self):
        self.queue.clear()
        self.converged = False

    def append(self, value):
        self.queue.append(value)

    def converge(self):
        if self.converged:
            return True
        else:
            losses = np.array(list(self.queue))
            self.converged = (len(losses) >= self.history_len) and (np.mean(losses[self.history_len // 2:]) > np.mean(
                losses[:self.history_len // 2]) - self.threshold)
            return self.converged


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, num_picker = 2, image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.num_picker = num_picker
        # the proprioceptive obs is stored as uint8
        obs_dtype = np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.picker_states = np.empty((capacity, num_picker), dtype=np.uint8)
        self.next_picker_states = np.empty((capacity, num_picker), dtype = np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, picker_state, action, reward, next_obs, next_picker_state, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.picker_states[self.idx], picker_state)
        np.copyto(self.next_picker_states[self.idx], next_picker_state)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        picker_states = torch.as_tensor(self.picker_states[idxs], device=self.device).float()
        next_picker_states = torch.as_tensor(self.next_picker_states[idxs], device=self.device).float()
        
        return obses, picker_states, actions, rewards, next_obses, next_picker_states, not_dones

    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)
        picker_states = torch.as_tensor(self.picker_states[idxs], device=self.device)
        next_picker_states = torch.as_tensor(self.next_picker_states[idxs], device=self.device)
        return obses, picker_states, actions, rewards, next_obses, next_picker_states, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.picker_states[self.last_save:self.idx],
            self.next_picker_states[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.picker_states[start:end] = payload[5]
            self.next_picker_states[start:end] = payload[6]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]
        picker_state = self.picker_states[idx]
        next_picker_state = self.next_picker_states[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, picker_state, action, reward, next_obs, next_picker_state, not_done

    def __len__(self):
        return self.capacity

class ReplayBufferExpert(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, num_picker = 2, image_size=84, transform=None):
        super().__init__(obs_shape, action_shape, capacity, batch_size, device, num_picker, image_size, transform)
        self.expert = np.empty((capacity, 1), dtype=np.float32)
        self.count_length = []
        self._expert_idx = []
    
    def add(self, obs, picker_state, action, reward, next_obs, next_picker_state, done, expert):
        while self.expert[self.idx]  == 1:
            self.idx = (self.idx + 1) % self.capacity
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.picker_states[self.idx], picker_state)
        np.copyto(self.next_picker_states[self.idx], next_picker_state)
        np.copyto(self.expert[self.idx], expert)
        if expert == 1:
            self._expert_idx.append(self.idx)
        if self.idx >= self.__len__():
            self.count_length.append(done)
        else:
            self.count_length[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def sample_proprio(self):
        assert len(self._expert_idx) >= self.batch_size/2
        assert len(self.__len__()) - len(self._expert_idx) >= self.batch_size/2
        expert_indexes = np.random.choice(self._expert_idx, size=int(self.batch_size/2)).tolist()
        non_expert_mask = np.ones(len(self.obses), dtype=bool)
        non_expert_mask[np.array(self._expert_idx)] = 0
        non_expert_indexes = np.random.choice(np.arange(len(self.obses))[non_expert_mask], size=int(self.batch_size/2)).tolist()
        indexes = expert_indexes + non_expert_indexes
        obses = self.obses[indexes]
        next_obses = self.next_obses[indexes]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[indexes], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[indexes], device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[indexes], device=self.device).float()
        picker_states = torch.as_tensor(self.picker_states[indexes], device=self.device).float()
        next_picker_states = torch.as_tensor(self.next_picker_states[indexes], device=self.device).float()
        expert = torch.as_tensor(self.expert[indexes], device=self.device).bool()
        return obses, picker_states, actions, rewards, next_obses, next_picker_states, not_dones, expert
    
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        patload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.picker_states[self.last_save:self.idx],
            self.next_picker_states[self.last_save:self.idx],
            self.expert[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(patload, path)
    
    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chunks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chunks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            patload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = patload[0]
            self.next_obses[start:end] = patload[1]
            self.actions[start:end] = patload[2]
            self.rewards[start:end] = patload[3]
            self.not_dones[start:end] = patload[4]
            self.picker_states[start:end] = patload[5]
            self.next_picker_states[start:end] = patload[6]
            self.expert[start:end] = patload[7]
            self.idx = end
    
    def __getitem__(self, idx):
        obs, picker_state, action, reward, next_obs, next_picker_state, not_done = super().__getitem__(idx)
        expert = self.expert[idx]
        return obs, picker_state, action, reward, next_obs, next_picker_state, not_done, expert
    
    def __len__(self):
        return len(self.count_length)


class PrioritizedReplayBufferExpert:
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, num_picker = 2, image_size=84, transform=None, alpha=0.6):
        self.buffer = ReplayBufferExpert(obs_shape, action_shape, capacity, batch_size, device, num_picker, image_size, transform)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, key):
        return self.buffer[key]
    
    def __setitem__(self, key, value):
        self.buffer[key] = value
    
    def add(self, obs, picker_state, action, reward, next_obs, next_picker_state, done, expert):
        idx = self.buffer.idx
        self.buffer.add(obs, picker_state, action, reward, next_obs, next_picker_state, done, expert)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
    def _sample_proportional(self, batch_size):
        res = []
        for i in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self.buffer) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self, beta=0.4):
        assert beta > 0
        idxes = self._sample_proportional(self.buffer.batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)
        # import ipdb; ipdb.set_trace()
        weights = torch.tensor(weights, dtype=torch.float, device=self.buffer.device)
        obses = self.buffer.obses[idxes]
        next_obses = self.buffer.next_obses[idxes]

        obses = torch.as_tensor(obses, device=self.buffer.device).float()
        actions = torch.as_tensor(self.buffer.actions[idxes], device=self.buffer.device).float()
        rewards = torch.as_tensor(self.buffer.rewards[idxes], device=self.buffer.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.buffer.device).float()
        not_dones = torch.as_tensor(self.buffer.not_dones[idxes], device=self.buffer.device).float()
        picker_states = torch.as_tensor(self.buffer.picker_states[idxes], device=self.buffer.device).float()
        next_picker_states = torch.as_tensor(self.buffer.next_picker_states[idxes], device=self.buffer.device).float()
        expert = torch.as_tensor(self.buffer.expert[idxes], device=self.buffer.device).bool()
        
        return obses, picker_states, actions, rewards, next_obses, next_picker_states, not_dones, expert, weights, idxes
    
    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class PrioritizedReplayBufferAugmented(PrioritizedReplayBufferExpert):
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, num_picker = 2, image_size=84, transform=None, alpha=0.6, aug_n=9):
        super().__init__(obs_shape, action_shape, capacity, batch_size, device, num_picker, image_size, transform, alpha)
        self.aug_n = aug_n

    def add(self, obs, picker_state, action, reward, next_obs, next_picker_state, done, expert):
        super().add(obs, picker_state, action, reward, next_obs, next_picker_state, done, expert)
        # os.makedirs('augmented', exist_ok=True)
        # plt.figure(figsize=(15, 15))
        # plt.subplot(self.aug_n+1, 2, 1)
        # plt.imshow(obs[0].numpy().transpose(1, 2, 0))
        # plt.subplot(self.aug_n+1, 2, 2)
        # plt.imshow(next_obs[0].numpy().transpose(1, 2, 0))

        for i in range(self.aug_n):
            obs_, action_, reward_, next_obs_, done_ = augmentTransition(obs, action, reward, next_obs, done, DEFAULT_CONFIG['aug_type'])
            # plt.subplot(self.aug_n+1, 2, 2*i+3)
            # plt.imshow(obs_[0].numpy().transpose(1, 2, 0))
            # plt.subplot(self.aug_n+1, 2, 2*i+4)
            # plt.imshow(next_obs_[0].numpy().transpose(1, 2, 0))
            super().add(obs_, picker_state, action_, reward_, next_obs_, next_picker_state, done_, expert)
        # plt.savefig(f'augmented/{self.buffer.idx}.png')
        # exit()

class ReplayBufferAugmented(ReplayBuffer):
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, num_picker = 2, image_size=84, transform=None, aug_n=9):
        super().__init__(obs_shape, action_shape, capacity, batch_size, device, num_picker, image_size, transform)
        self.aug_n = aug_n
    
    def add(self, obs, picker_state, action, reward, next_obs, next_picker_state, done):
        super().add(obs, picker_state, action, reward, next_obs, next_picker_state, done)
        # os.makedirs('augmented', exist_ok=True)
        # plt.figure(figsize=(15, 15))
        # plt.subplot(self.aug_n+1, 2, 1)
        # plt.imshow(obs[0].numpy().transpose(1, 2, 0))
        # plt.subplot(self.aug_n+1, 2, 2)
        # plt.imshow(next_obs[0].numpy().transpose(1, 2, 0))

        for i in range(self.aug_n):
            obs_, action_, reward_, next_obs_, done_ = augmentTransition(obs, action, reward, next_obs, done, DEFAULT_CONFIG['aug_type'])
            # plt.subplot(self.aug_n+1, 2, 2*i+3)
            # plt.imshow(obs_[0].numpy().transpose(1, 2, 0))
            # plt.subplot(self.aug_n+1, 2, 2*i+4)
            # plt.imshow(next_obs_[0].numpy().transpose(1, 2, 0))    
            super().add(obs_, picker_state, action_, reward_, next_obs_, next_picker_state ,done_)
        # plt.savefig(f'augmented/{self.idx}.png')
        # exit()

def augmentTransition(obs, action, reward, next_obs, done, aug_type):
    if aug_type=='se2':
        return augmentTransitionSE2(obs, action, reward, next_obs, done)
    elif aug_type=='so2':
        return augmentTransitionSO2(obs, action, reward, next_obs, done)
    elif aug_type=='trans':
        return augmentTransitionTranslate(obs, action, reward, next_obs, done)
    elif aug_type=='shift':
        return augmentTransitionShift(obs, action, reward, next_obs, done)
    else:
        raise NotImplementedError

def augmentTransitionSE2(obs, action, reward, next_obs, done):
    dxy = action[::2].copy()
    dxy1, dxy2 = np.split(dxy, 2)
    obs, next_obs, dxy1, dxy2, transform_params = perturb(obs[0].numpy().copy(),
                                                          next_obs[0].numpy().copy(),
                                                          dxy1, dxy2)
    obs = obs.reshape(1, *obs.shape)
    obs = torch.from_numpy(obs)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    next_obs = torch.from_numpy(next_obs)
    action = action.copy()
    action[0] = dxy1[0]
    action[2] = dxy1[1]
    action[4] = dxy2[0]
    action[6] = dxy2[1]
    return obs, action, reward, next_obs, done
    
def augmentTransitionSO2(obs, action, reward, next_obs, done):
    dxy = action[::2].copy()
    dxy1, dxy2 = np.split(dxy, 2)
    obs, next_obs, dxy1, dxy2, transform_params = perturb(obs[0].numpy().copy(),
                                                          next_obs[0].numpy().copy(),
                                                          dxy1, dxy2,
                                                          set_trans_zero=True)
    obs = obs.reshape(1, *obs.shape)
    obs = torch.from_numpy(obs)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    next_obs = torch.from_numpy(next_obs)
    action = action.copy()
    action[0] = dxy1[0]
    action[2] = dxy1[1]
    action[4] = dxy2[0]
    action[6] = dxy2[1]
    return obs, action, reward, next_obs, done

def augmentTransitionTranslate(obs, action, reward, next_obs, done):
    dxy = action[::2].copy()
    dxy1, dxy2 = np.split(dxy, 2)
    obs, next_obs, dxy1, dxy2, transform_params = perturb(obs[0].numpy().copy(),
                                                          next_obs[0].numpy().copy(),
                                                          dxy1, dxy2,
                                                          set_theta_zero=True)
    obs = obs.reshape(1, *obs.shape)
    obs = torch.from_numpy(obs)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    next_obs = torch.from_numpy(next_obs)
    return obs, action, reward, next_obs, done

def augmentTransitionShift(obs, action, reward, next_obs, done):
    heightmap_size = obs.shape[-1]
    padded_obs = transforms.Pad((4, 4, 4, 4), padding_mode='edge')(obs)
    padded_next_obs = transforms.Pad((4, 4, 4, 4), padding_mode='edge')(next_obs)
    mag_x = np.random.randint(8)
    mag_y = np.random.randint(8)
    obs = padded_obs[:, :, mag_x:mag_x+heightmap_size, mag_y:mag_y+heightmap_size]
    next_obs = padded_next_obs[:, :, mag_x:mag_x+heightmap_size, mag_y:mag_y+heightmap_size]
    return obs, action, reward, next_obs, done

def perturb(current_image, next_image, dxy1, dxy2, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]
    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]])
    rotated_dxy1 = rot.dot(dxy1)
    rotated_dxy1 = np.clip(rotated_dxy1, -1, 1)
    
    rotated_dxy2 = rot.dot(dxy2)
    rotated_dxy2 = np.clip(rotated_dxy2, -1, 1)
    # import ipdb; ipdb.set_trace()
    # Apply rigid transform to image and pixel labels.
    if current_image.shape[0] == 1:
        current_image = affine_transform(current_image[0], np.linalg.inv(transform), mode='nearest', order=1).reshape(current_image.shape)
        if next_image is not None:
            next_image = affine_transform(next_image[0], np.linalg.inv(transform), mode='nearest', order=1).reshape(next_image.shape)
    else:
        for i in range(current_image.shape[0]):
            current_image[i, :, :] = affine_transform(current_image[i, :, :], np.linalg.inv(transform), mode='nearest', order=1)
            if next_image is not None:
                next_image[i, :, :] = affine_transform(next_image[i, :, :], np.linalg.inv(transform), mode='nearest', order=1)
            if i >= 3:
                break
    return current_image, next_image, rotated_dxy1, rotated_dxy2, transform_params


def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], 
                              [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], 
                              [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], 
                          [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    # print('input image shape:', image.shape)
    if image.shape[0] ==1:
        image = image[0]
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    # print('output image shape:', image.shape)
    return image

def choose_random_particle_from_boundary(env):
    picker_pos, particle_pos = env.action_tool._get_pos()
    # take the corner points
    corner_point_upper_left = env._get_key_point_idx()[0]
    corner_point_lower_left = env._get_key_point_idx()[1]
    corner_point_upper_right = env._get_key_point_idx()[2]
    corner_point_lower_right = env._get_key_point_idx()[3]

    # check whether a  point above the corner point
    corner_upper_left_flag = True
    corner_lower_left_flag = True
    corner_upper_right_flag = True
    corner_lower_right_flag = True
    for i in range(len(particle_pos)):
        if particle_pos[i, 1] > particle_pos[corner_point_lower_left, 1] and np.linalg.norm(particle_pos[i, [0, 2]] - particle_pos[corner_point_lower_left, [0, 2]]) < 2 * env.cloth_particle_radius:
            corner_lower_left_flag = False
        if particle_pos[i, 1] > particle_pos[corner_point_upper_left, 1] and np.linalg.norm(particle_pos[i, [0, 2]] - particle_pos[corner_point_upper_left, [0, 2]]) < 2 * env.cloth_particle_radius:
            corner_upper_left_flag = False
        if particle_pos[i, 1] > particle_pos[corner_point_lower_right, 1] and np.linalg.norm(particle_pos[i, [0, 2]] - particle_pos[corner_point_lower_right, [0, 2]]) < 2 * env.cloth_particle_radius:
            corner_lower_right_flag = False
        if particle_pos[i, 1] > particle_pos[corner_point_upper_right, 1] and np.linalg.norm(particle_pos[i, [0, 2]] - particle_pos[corner_point_upper_right, [0, 2]]) < 2 * env.cloth_particle_radius:
            corner_upper_right_flag = False

    # import ipdb; ipdb.set_trace()
    cloth_size = env.current_config['ClothSize']

    corner = []
    if cloth_size[0] > cloth_size[1]:
        if corner_upper_left_flag and corner_upper_right_flag:
            corner.append((corner_point_upper_left, corner_point_upper_right))
        if corner_lower_left_flag and corner_lower_right_flag:
            corner.append((corner_point_lower_left, corner_point_lower_right))
    else:
        if corner_upper_left_flag and corner_lower_left_flag:
            corner.append((corner_point_upper_left, corner_point_lower_left))
        if corner_upper_right_flag and corner_lower_right_flag:
            corner.append((corner_point_upper_right, corner_point_lower_right))
    if len(corner) > 0:
        choosen_id = np.array(random.sample(corner, 1)[0])
    else:
        corner_se = []
        if cloth_size[0] > cloth_size[1]:
            if corner_upper_left_flag and corner_lower_left_flag:
                corner_se.append((corner_point_upper_left, corner_point_lower_left))
            if corner_upper_right_flag and corner_lower_right_flag:
                corner_se.append((corner_point_upper_right, corner_point_lower_right))
        else:
            if corner_upper_left_flag and corner_upper_right_flag:
                corner_se.append((corner_point_upper_left, corner_point_upper_right))
            if corner_lower_left_flag and corner_lower_right_flag:
                corner_se.append((corner_point_lower_left, corner_point_lower_right))
        if len(corner_se) > 0:
            choosen_id = np.array(random.sample(corner_se, 1)[0])
        else:
            try:
                hull = ConvexHull(particle_pos[:, [0, 2]])
            except:
                return None

            bound_id = set()
            for simplex in hull.simplices:
                bound_id.add(simplex[0])
                bound_id.add(simplex[1])

            single_corner = []
            if corner_upper_left_flag:
                single_corner.append(corner_point_upper_left)
            if corner_lower_left_flag:
                single_corner.append(corner_point_lower_left)
            if corner_upper_right_flag: 
                single_corner.append(corner_point_upper_right)
            if corner_lower_right_flag:
                single_corner.append(corner_point_lower_right)
            if len(single_corner) > 0:
                _id_1 = random.sample(single_corner, 1)
                count_single_id = 0
                while True:
                    _id_2 = random.sample(bound_id, 1)
                    count_single_id += 1
                    if _id_1 != _id_2 and abs(particle_pos[_id_1, 0] - particle_pos[_id_2, 0]) >=  4 * env.action_tool.picker_radius:
                        choosen_id = np.array([_id_1[0], _id_2[0]])
                        break
                    if count_single_id > 20:
                        return None
            else:
                count_choose_id = 0
                while True:
                    choosen_id = random.sample(bound_id, 2)
                    count_choose_id += 1
                    if abs(particle_pos[choosen_id[0], 0] - particle_pos[choosen_id[1], 0]) >=  4 * env.action_tool.picker_radius:
                        break
                    if count_choose_id > 20:
                        return None
    # find the closest points for picker
    if particle_pos[choosen_id[0], 0] > particle_pos[choosen_id[1], 0]:
        return np.array([choosen_id[1], choosen_id[0]])
    return choosen_id

def pick_choosen_point(env, obs, picker_state, choosen_id, thresh, episode_step, frames, expert_data, max_step=20, img_size=128):
    count_pick_bound = 0
    while True:
        picker_pos, particle_pos = env.action_tool._get_pos()
        target_pos = particle_pos[choosen_id, :3]
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        if (norm <= thresh).all():
            action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
        else:
            action = np.concatenate([action, np.zeros((2, 1))], axis=1).reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        count_pick_bound += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon or count_pick_bound >= max_step:
            return None
        if np.all(picker_state == 1) and len(set(particle_pos[env.action_tool.picked_particles, 3])) == 1:
            return [episode_step, obs, picker_state]

def fling_primitive_1(env, obs, picker_state, choosen_id, thresh, episode_step, frames, expert_data, final_step, img_size=128):
    episode_step_offset = episode_step
    # first move the height 0.3, keep the distance between 2 particles
    curr_pos = env.action_tool._get_pos()[0]
    dist = abs(curr_pos[0, 0] - curr_pos[1, 0])
    target_pos = np.array([[-dist/2, 0.3, -0.1], [dist/2, 0.3, -0.1]])
    while True:
        picker_pos = env.action_tool._get_pos()[0]
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon:
            return None
        if (norm < thresh).all():
            break

    # second, stretch the cloth
    init_pos = env._get_flat_pos()
    init_dis = np.linalg.norm(init_pos[choosen_id[0], [0, 2]] - init_pos[choosen_id[1], [0, 2]])
    curr_pos, particle_pos = env.action_tool._get_pos()
    curr_dis = np.linalg.norm(particle_pos[choosen_id[0], [0, 2]] - particle_pos[choosen_id[1], [0, 2]])
    denta = (init_dis-curr_dis) / 2
    if curr_pos[0, 0] > curr_pos[1, 0]:
        left = 1
        right = 0
    else:
        left = 0
        right = 1
    cos_phi = (curr_pos[right, 0] - curr_pos[left, 0]) / curr_dis
    sin_phi = (curr_pos[right, 2] - curr_pos[left, 2]) / curr_dis
    target_pos = copy.deepcopy(curr_pos)
    target_pos[left, 0] = curr_pos[left, 0] - denta * cos_phi
    target_pos[left, 2] = curr_pos[left, 2] - denta * sin_phi
    target_pos[right, 0] = curr_pos[right, 0] + denta * cos_phi
    target_pos[right, 2] = curr_pos[right, 2] + denta * sin_phi
    while True:
        picker_pos = env.action_tool._get_pos()[0]
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon:
            return None
        if (norm < thresh).all():
            break
    final_step.append(episode_step-episode_step_offset)

    # third, lift until one particle on the ground
    for _ in range(20):
        action = np.array([0.0, 0.5, 0.0, 1, 0.0, 0.5, 0.0, 1])
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon:
            return None
        if (env.action_tool._get_pos()[1][:, 1] >= 2*env.cloth_particle_radius).all():
            break

    for _ in range (4):
        action = np.array([0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 1])
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
    final_step.append(episode_step-episode_step_offset)
    # last, fling the cloth towards
    current_pos = env.action_tool._get_pos()[0]
    fling_height = current_pos[0, 1]
    if fling_height <= 0.55:
        count_loop = 4
    elif fling_height <= 0.7:
        count_loop = 7
    else:
        count_loop = 10

    for i in range (count_loop):
        action = np.array([0.0, 0.0, np.exp(-1*(i+1)/count_loop), 1, 0.0, 0.0, np.exp(-1*(i+1)/count_loop), 1])
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon:
            return None

    # next, move back the cloth to the ground
    curr_pos = env.action_tool._get_pos()[0]
    target_pos = copy.deepcopy(curr_pos)
    target_pos[:, 1] = 0.04
    target_pos[:, 2] = -0.4
    while True:
        picker_pos = env.action_tool._get_pos()[0]
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon:
            return None
        if (norm < thresh).all():
            break

    # last, slowly move the cloth to the ground
    curr_pos = env.action_tool._get_pos()[0]
    target_pos = copy.deepcopy(curr_pos)
    target_pos[:, 2] -= 1.5
    for _ in range(3):
        picker_pos = env.action_tool._get_pos()[0]
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.ones((2, 1))], axis=1).reshape(-1)
        next_obs, reward, done, info = env.step(action)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        episode_step += 1
        obs = next_obs
        picker_state = next_picker_state
        if episode_step == env.horizon:
            return None

    # set the grasp is False
    action = np.array([-1.0, 1.0, 0.0, 0, 1.0, 1.0, 0.0, 0])
    next_obs, reward, done, info = env.step(action)
    next_picker_state = get_picker_state(env)
    expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
    frames.append(env.get_image(img_size, img_size))
    episode_step += 1
    obs = next_obs
    picker_state = next_picker_state
    if done:
        final_step.append(episode_step-episode_step_offset)
        return 1
    if episode_step == env.horizon:
        return None
    return [episode_step, obs, picker_state]

def pick_and_drag(env, thresh, img_size=128):
    obs = env.reset()
    frames = []
    expert_data = []
    env._set_to_flatten()
    frames.append(env.get_image(img_size, img_size))
    picker_state = get_picker_state(env)

    picker_pos, particle_pos = env.action_tool._get_pos()
    corner = env._get_key_point_idx()
    rand_id = np.random.randint(0, 4)
    choosen_id = corner[rand_id]
    target_pos = picker_pos
    if particle_pos[choosen_id, 0] >= 0:
        target_pos[1, :] = particle_pos[choosen_id, :3]
        left = 0
    else:
        target_pos[0, :] = particle_pos[choosen_id, :3]
        left = 1
    
    count = 0
    while True:
        count += 1
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        if (norm <= thresh).all():
            if left:
                action[0, -1] = 1
            else:
                action[1, -1] = 1
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        if left and picker_state[0] == 1:
            break
        if not left and picker_state[1] == 1:
            break
        if count >= 10:
            break
    
    # lift a little bit
    for _ in range(3):
        if left:
            action = np.array([0.0, 0.3, 0.0, 1, 0.0, 0.0, 0.0, 0])
        else:
            action = np.array([0.0, 0.0, 0.0, 0, 0.0, 0.3, 0.0, 1])
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state

    # choose one random point in particle_pos
    picker_pos, particle_pos = env.action_tool._get_pos()
    while True:
        choosen_id_ = np.random.randint(0, len(particle_pos))
        if np.linalg.norm(particle_pos[choosen_id_, [0, 2]] - picker_pos[0, [0, 2]]) <= 0.3 and np.linalg.norm(particle_pos[choosen_id_, [0, 2]] - picker_pos[0, [0, 2]]) >= 0.1 and left:
            break
        if np.linalg.norm(particle_pos[choosen_id_, [0, 2]] - picker_pos[1, [0, 2]]) <= 0.3 and np.linalg.norm(particle_pos[choosen_id_, [0, 2]] - picker_pos[1, [0, 2]]) >= 0.1 and not left:
            break
    # move to the choosen point
    target_pos = picker_pos
    if left:
        target_pos[0, :] = particle_pos[choosen_id_, :3]
        target_pos[0, 1] += 0.04
    else:
        target_pos[1, :] = particle_pos[choosen_id_, :3]
        target_pos[1, 1] += 0.04
    count = 0
    while True:
        count += 1
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.02, 0.02) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        if left:
            action[0, -1] = 1
        else:
            action[1, -1] = 1
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        if (norm <= thresh).all():
            break
    
    for _ in range(2):
        if left:
            action = np.array([0.0, -0.25, 0.0, 1, 0.0, 0.0, 0.0, 0])
        else:
            action = np.array([0.0, 0.0, 0.0, 0, 0.0, -0.25, 0.0, 1])
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs

    # throw the cloth
    if left:
        action = np.array([0.0, 1.0, 0.0, 0, 0.0, 0.0, 0.0, 0])
    else:
        action = np.array([0.0, 0.0, 0.0, 0, 0.0, 1.0, 0.0, 0])
    next_obs, reward, done, info = env.step(action)
    next_picker_state = get_picker_state(env)
    expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
    frames.append(env.get_image(img_size, img_size))
    obs = next_obs
    picker_state = next_picker_state

    picker_pos, particle_pos = env.action_tool._get_pos()
    target_pos = picker_pos
    random_y = np.random.uniform(-0.6, 0.6)
    random_z = np.random.uniform(0.2, 0.75)
    if left:
        random_x = np.random.uniform(-0.6, 0)
        target_pos[0, :] = [random_x, random_z, random_y]
    else:
        random_x = np.random.uniform(0, 0.6)
        target_pos[1, :] = [random_x, random_z, random_y]
    while True:
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        if (norm <= thresh).all():
            break

    # take the action
    acts = []
    for i in reversed(expert_data):
        ps = i[5]
        next_ps = i[6]
        act = i[1]
        act = act.reshape(2, -1)
        act[:, :3] *= -1
        denta = np.empty((2, 1))
        for j in range(2):
            if ps[j] == 0 and next_ps[j] == 0:
                denta[j] = 0
            elif ps[j] == 0 and next_ps[j] == 1:
                denta[j] = 0
            elif ps[j] == 1 and next_ps[j] == 0:
                denta[j] = 1
            elif ps[j] == 1 and next_ps[j] == 1:
                denta[j] = 1
        act[:, 3] = denta.reshape(2,)
        act = act.reshape(-1)
        acts.append(act)
    
    new_frames = [env.get_image(img_size, img_size)]
    picker_state = np.array([0, 0])
    new_expert_data = []
    final_step = []
    episode_step = 0

    for i in range(len(acts)):
        action = acts[i]
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        new_frames.append(env.get_image(img_size, img_size))
        new_expert_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        episode_step += 1
        if (left and picker_state[0] == 0 and next_picker_state[0] == 1 and done) or (not left and picker_state[1] == 0 and next_picker_state[1] == 1 and done):
            final_step.append(episode_step-1)
        if (left and picker_state[0] == 1 and next_picker_state[0] == 0 and done) or (not left and picker_state[1] == 1 and next_picker_state[1] == 0 and done):
            final_step.append(episode_step)
            return [1, final_step, new_frames, new_expert_data]
        obs = next_obs
        picker_state = next_picker_state
    return None

def pick_and_drag_play(env, obs, frames, play_data, thresh, img_size=128):
    picker_state = get_picker_state(env)
    picker_pos, particle_pos = env.action_tool._get_pos()
    random_id = np.random.randint(0, len(particle_pos))
    target_pos = picker_pos
    if particle_pos[random_id, 0] >= 0:
        left = 0
        target_pos[1, :] = particle_pos[random_id, :3]
    else:
        left = 1
        target_pos[0, :] = particle_pos[random_id, :3]
    
    # move to the target position
    count = 0
    while True:
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        if (norm <= thresh).all() or count >= 10:
            if left:
                action[0, -1] = 1
            else:
                action[1, -1] = 1
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        count += 1
        if left and picker_state[0] == 1:
            break
        if not left and picker_state[1] == 1:
            break
    
    # random a position (x, y) uniform in range [-0.6, 0.6]
    picker_pos, particle_pos = env.action_tool._get_pos()
    random_x = np.random.uniform(-0.6, 0.6)
    random_y = np.random.uniform(-0.6, 0.6)
    dxy = np.sqrt(random_x**2 + random_y**2)
    random_z = np.random.uniform(0.1, min(1.1-dxy, 0.75))
    target_pos = picker_pos
    if left:
        target_pos[0] = [random_x, random_z, random_y]
    else:
        target_pos[1] = [random_x, random_z, random_y]

    # move to the target position
    while True:
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.04, 0.04) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        if left:
            action[0, -1] = 1
        else:
            action[1, -1] = 1
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        if (norm <= thresh).all():
            break

    action = np.array([0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0])
    next_obs, reward, done, info = env.step(action)
    next_picker_state = get_picker_state(env)
    play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
    frames.append(env.get_image(img_size, img_size))
    obs = next_obs
    picker_state = next_picker_state

    obs, frames, play_data = random_picker_position(env, obs, frames, play_data, thresh, img_size)
    # save_numpy_as_gif(np.array(frames), 'test.gif')
    return obs, frames, play_data

def random_picker_position(env, obs, frames, play_data, thresh, img_size=128):
    picker_state = get_picker_state(env)
    picker_pos, _ = env.action_tool._get_pos()
    if np.random.uniform(0.0, 1.0) <= 0.5:
        # random a position (x, y) uniform in range [-0.6, 0.6]
        random_x_1 = np.random.uniform(-0.6, 0.6)
        random_y_1 = np.random.uniform(-0.6, 0.6)
        dxy = np.sqrt(random_x_1**2 + random_y_1**2)
        random_z_1 = np.random.uniform(0.2, min(1.1-dxy, 0.75))

        random_x_2 = np.random.uniform(-0.6, 0.6)
        random_y_2 = np.random.uniform(-0.6, 0.6)
        dxy = np.sqrt(random_x_2**2 + random_y_2**2)
        random_z_2 = np.random.uniform(0.2, min(1.1-dxy, 0.75))

        if random_x_1 > random_x_2:
            random_x_1, random_x_2 = random_x_2, random_x_1
        target_pos = np.array([[random_x_1, random_z_1, random_y_1], [random_x_2, random_z_2, random_y_2]])
    else:
        target_pos = np.array([[-0.3, 0.3, 0.0], [0.3, 0.3, 0.0]])
    while True:
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        if (norm <= thresh).all():
            break
    return obs, frames, play_data


def pick_by_2_picker_and_drag(env, obs, frames, play_data, thresh, img_size=128):
    picker_state = get_picker_state(env)
    picker_pos, particle_pos = env.action_tool._get_pos()
    random_id_1 = np.random.randint(0, len(particle_pos))
    random_id_2 = np.random.randint(0, len(particle_pos))
    while random_id_1 == random_id_2 or np.linalg.norm(particle_pos[random_id_1, :3] - particle_pos[random_id_2, :3]) <= 0.1:
        random_id_2 = np.random.randint(0, len(particle_pos))
    
    target_pos1 = particle_pos[random_id_1, :3]
    target_pos2 = particle_pos[random_id_2, :3]
    if target_pos1[0] >= target_pos2[0]:
        target_pos = np.concatenate([target_pos2.reshape(1, -1), target_pos1.reshape(1, -1)], axis=0)
    else:
        target_pos = np.concatenate([target_pos1.reshape(1, -1), target_pos2.reshape(1, -1)], axis=0)
    count = 0
    while True:
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.08, 0.08) / 0.08
        action = np.concatenate([action, np.zeros((2, 1))], axis=1)
        if (norm <= thresh).all() or count >= 20:
            action[:, -1] = 1
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        count += 1
        if (picker_state == 1).all() or count >= 20:
            break
    
    # initial position
    init_pos = env._get_flat_pos()
    init_dis = np.linalg.norm(init_pos[random_id_1, :3] - init_pos[random_id_2, :3])
    # random a position (x, y) uniform in range [-0.6, 0.6]
    picker_pos, particle_pos = env.action_tool._get_pos()
    random_x1 = np.random.uniform(-0.6, 0.6)
    random_y1 = np.random.uniform(-0.6, 0.6)
    dxy = np.sqrt(random_x1**2 + random_y1**2)
    random_z1 = np.random.uniform(0.2, min(1.1-dxy, 0.75))

    denta = np.random.uniform(0.0, init_dis)
    r = np.random.uniform(0.0, 1.0)
    if r <= 0.33:
        random_x2 = random_x1 + denta
        random_y2 = random_y1
        random_z2 = random_z1
    elif r <= 0.66:
        random_x2 = random_x1
        random_y2 = random_y1 + denta
        random_z2 = random_z1
    else:
        random_x2 = random_x1
        random_y2 = random_y1
        random_z2 = random_z1 + denta

    if random_x1 > random_x2:
        random_x1, random_x2 = random_x2, random_x1
    target_pos = np.array([[random_x1, random_z1, random_y1], [random_x2, random_z2, random_y2]])
    count = 0
    while True:
        picker_pos, _ = env.action_tool._get_pos()
        dis = target_pos - picker_pos
        norm = np.linalg.norm(dis, axis=1)
        action = np.clip(dis, -0.04, 0.04) / 0.08
        action = np.concatenate([action, np.ones((2, 1))], axis=1)
        action = action.reshape(-1)
        next_obs, reward, done, info = env.step(action)
        next_picker_state = get_picker_state(env)
        play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
        frames.append(env.get_image(img_size, img_size))
        obs = next_obs
        picker_state = next_picker_state
        count += 1
        if (norm <= thresh).all() or count >= 20:
            break

    action = np.array([0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0])
    next_obs, reward, done, info = env.step(action)
    next_picker_state = get_picker_state(env)
    play_data.append([obs, action, reward, next_obs, float(done), picker_state, next_picker_state])
    frames.append(env.get_image(img_size, img_size))
    obs = next_obs
    picker_state = next_picker_state

    obs, frames, play_data = random_picker_position(env, obs, frames, play_data, thresh, img_size)
    # save_numpy_as_gif(np.array(frames), 'test.gif')
    return obs, frames, play_data


def create_demonstration(env,
                         video_dir,
                         num_demonstrations,
                         img_size=128,
):
    print(f'[INFO] =================================================================================================')
    print(f'[INFO] ==================== START COLLECTING {num_demonstrations} DEMONSTRATIONS AND PLAY DATA ====================')
    print(f'[INFO] {num_demonstrations//2} for pick and drag, {num_demonstrations//2} for fling')
    print(f'[INFO] AND collecting 100 play data')
    print(f'[INFO] =================================================================================================')
    
    # create directory to save demonstratio
    demo_npy = os.path.join(video_dir, 'demo_npy')
    if not os.path.exists(demo_npy):
        os.makedirs(demo_npy, exist_ok=True)

    play_npy = os.path.join(video_dir, 'play_npy')
    if not os.path.exists(play_npy):
        os.makedirs(play_npy, exist_ok=True)

    all_frames_planner = []
    all_expert_data_planner = []
    all_play_data_planner = []
    all_play_frames_planner = []
    count_expert_planner = 0
    count_play_planner = 0
    Demo_Length = []
    Demo_Final_Step = []
    Demo_NPY = []
    PLAY_NPY = []

    thresh = env.cloth_particle_radius + env.action_tool.picker_radius + env.action_tool.picker_threshold
    while True:
        obs = env.reset()
        play_data = []
        play_frames = [env.get_image(img_size, img_size)]
        if np.random.uniform(0.0, 1.0) <= 0.5:
            # reinitialize the cloth and picker
            obs, play_frames, play_data = pick_and_drag_play(env, obs, play_frames, play_data, thresh, img_size=img_size)
        episode_step = len(play_data) if len(play_data) != 0 else 0
        flag_reset = False
        while True:
            final_step = []
            expert_data = []
            frames = [env.get_image(img_size, img_size)]
            picker_state = get_picker_state(env)
            episode_step_offset = episode_step
            # choose random boundary point
            choosen_id = choose_random_particle_from_boundary(env)
            if choosen_id is None:
                print('[INFO] Cannot find boundary point!!!')
                flag_reset = True
                break
            # move to two choosen boundary points and pick them
            pick_choosen = pick_choosen_point(env, obs, picker_state, choosen_id, thresh, episode_step, frames, expert_data, img_size=img_size)
            if pick_choosen is None:
                flag_reset = True
                break
            else:
                episode_step, obs, picker_state = pick_choosen[0], pick_choosen[1], pick_choosen[2]
                final_step.append(episode_step-1-episode_step_offset)
            # fling primitive
            fling = fling_primitive_1(env, obs, picker_state, choosen_id, thresh, episode_step, frames, expert_data, final_step, img_size=img_size)
            if fling == 1:
                for i in range(len(final_step)):
                    if i != 0:
                        final_step[i] += final_step[0] + 1
                assert final_step[-1] == len(frames) - 1
                count_expert_planner += 1
                break
            if fling is None:
                flag_reset = True
                break
            episode_step, obs, picker_state = fling[0], fling[1], fling[2]
            # copy data in list frames and expert_data to play_frames and play_data
            for i in range(len(expert_data)):
                play_frames.append(frames[i+1])
                play_data.append(expert_data[i])
            # random picker position
            pre = len(play_data)
            obs, play_frames, play_data = random_picker_position(env, obs, play_frames, play_data, thresh, img_size=img_size)
            pos = len(play_data)
            episode_step += pos - pre
        if flag_reset:
            # copy data in list frames and expert_data to play_frames and play_data
            if len(expert_data) + len(play_data) == env.horizon:
                for i in range(len(expert_data)):
                    play_frames.append(frames[i+1])
                    play_data.append(expert_data[i])
                all_play_data_planner.append(play_data)
                count_play_planner += 1
                play_data_path = os.path.join(play_npy, f'data_{count_play_planner}.npy')
                np.save(play_data_path, play_data)
                PLAY_NPY.append(os.path.abspath(play_data_path))
            continue
        
        Demo_Final_Step.append(final_step)
        Demo_Length.append(len(frames))
       
        # save expert data type list to npy file
        expert_data_path = os.path.join(demo_npy, f'data_{count_expert_planner}.npy')
        np.save(expert_data_path, expert_data)
        Demo_NPY.append(os.path.abspath(expert_data_path))

        if len(frames) != env.horizon + 1:
            for _ in range(env.horizon + 1 - len(frames)):
                frames.append(env.get_image(img_size, img_size))
 
        all_frames_planner.append(frames)
        all_expert_data_planner.append(expert_data)
        print('[INFO]Collected {} fling demonstrations in {} steps'.format(count_expert_planner, len(expert_data)))
        if count_expert_planner == num_demonstrations//2:
            print('==================== FINISH COLLECTING FLING DEMONSTRATIONS ====================')
            break
    
    while True:
        pick_and_drag_out = pick_and_drag(env, thresh, img_size=img_size)
        if pick_and_drag_out is not None:
            count_expert_planner += 1
            final_step, frames, expert_data = pick_and_drag_out[1], pick_and_drag_out[2], pick_and_drag_out[3]
            Demo_Final_Step.append(final_step)
            Demo_Length.append(len(frames))

            # save expert data type list to npy file
            expert_data_path = os.path.join(demo_npy, f'data_{count_expert_planner}.npy')
            np.save(expert_data_path, expert_data)
            Demo_NPY.append(os.path.abspath(expert_data_path))

            if len(frames) != env.horizon + 1:
                for _ in range(env.horizon + 1 - len(frames)):
                    frames.append(env.get_image(img_size, img_size))

            all_frames_planner.append(frames)
            all_expert_data_planner.append(expert_data)
            print('[INFO]Collected {} pick and drag demonstrations in {} steps'.format(count_expert_planner-num_demonstrations//2, len(expert_data)))
            if count_expert_planner == num_demonstrations:
                print('==================== FINISH COLLECTING PICK AND DRAG DEMONSTRATIONS ====================')
                break

    df = pd.DataFrame({'Length': Demo_Length, 'Final_step': Demo_Final_Step, 'NPY_Path': Demo_NPY})
    df.to_csv(os.path.join(video_dir, 'demo.csv'), index=False)

    for i in range(num_demonstrations//20):
        sub_all_frames_planner = all_frames_planner[i*20:(i+1)*20] 
        sub_all_frames_planner = np.array(sub_all_frames_planner).swapaxes(0, 1)
        sub_all_frames_planner = np.array([make_grid(np.array(frame), nrow=4, padding=3) for frame in sub_all_frames_planner])
        save_numpy_as_gif(sub_all_frames_planner, os.path.join(video_dir, 'expert_{}.gif'.format(i)))
    
    
    for _ in range(100):
        obs = env.reset()
        play_data = []
        play_frames = [env.get_image(img_size, img_size)]
        while True:
            if np.random.uniform(0.0, 1.0) <= 0.5:
                obs, play_frames, play_data = pick_and_drag_play(env, obs, play_frames, play_data, thresh, img_size=img_size)
                obs, play_frames, play_data = pick_by_2_picker_and_drag(env, obs, play_frames, play_data, thresh, img_size=img_size)
            else:
                obs, play_frames, play_data = pick_by_2_picker_and_drag(env, obs, play_frames, play_data, thresh, img_size=img_size)
                obs, play_frames, play_data = pick_and_drag_play(env, obs, play_frames, play_data, thresh, img_size=img_size)
            if len(play_data) >= env.horizon + 1:
                break
        play_data = play_data[:env.horizon]
        play_frames = play_frames[:env.horizon+1]
        all_play_data_planner.append(play_data)
        count_play_planner += 1
        play_data_path = os.path.join(play_npy, f'data_{count_play_planner}.npy')
        np.save(play_data_path, play_data)
        PLAY_NPY.append(os.path.abspath(play_data_path))
        print('[INFO]Collected {} play data'.format(count_play_planner))

    # play_df = pd.DataFrame({'NPY_Path': PLAY_NPY})
    # play_df.to_csv(os.path.join(video_dir, 'play.csv'), index=False)
    # for i in range(100//20):
    #     sub_all_frames_planner = all_play_data_planner[i*20:(i+1)*20] 
    #     sub_all_frames_planner = np.array(sub_all_frames_planner).swapaxes(0, 1)
    #     sub_all_frames_planner = np.array([make_grid(np.array(frame), nrow=4, padding=3) for frame in sub_all_frames_planner])
    #     save_numpy_as_gif(sub_all_frames_planner, os.path.join(video_dir, 'play_{}.gif'.format(i)))
    
    return all_expert_data_planner, all_play_data_planner




def get_picker_state(env):
    picker_state = []
    for ps in env.action_tool.picked_particles:
        if ps is None:
            picker_state.append(0)
        else:
            picker_state.append(1) 
    return np.array(picker_state, dtype=np.uint8)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        if self.schedule_timesteps == 0:
            return self.final_p
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def predict_action(x, embs_concat, actions_concat, k=5):
    # find k nearest neighbors in embs_concat
    dists = np.linalg.norm(embs_concat - x, axis=-1)
    nn_ids = np.argsort(dists)[:k]
    nn_actions = actions_concat[nn_ids]
    nn_dist = dists[nn_ids]
    nn_dist = np.exp(-nn_dist)
    nn_dist = nn_dist / np.sum(nn_dist)

    for i in range(k):
        if i == 0:
            action = nn_dist[i] * nn_actions[i]
        else:
            action += nn_dist[i] * nn_actions[i]
    action[3] = 0 if action[3] <= 0.5 else 1
    action[7] = 0 if action[7] <= 0.5 else 1
    return action

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def load_model_byol(model, id=50):
    print(f'[INFO] Loading model with {id} ...')
    ckpt = torch.load(f'/home/hnguyen/cloth_smoothing/tcc/byol_{id}.pt', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    # model.fc = Identity()
    model.fc = nn.Identity()
    print('[INFO] Loaded model ...')
    return model