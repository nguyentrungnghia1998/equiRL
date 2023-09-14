import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from byol import BYOL, RandomChangeBackgroundRGBD, RandomGrayScaleRGBD, RandomColorJitterRGBD
from tqdm import tqdm
from behavior_transformer import BehaviorTransformer, GPT, GPTConfig
from utils import get_image_transform, get_random_image_transform_params
from scipy.ndimage import affine_transform
import argparse

def load_model(model, path='/home/hnguyen/cloth_smoothing/equiRL/models/pretrained_observation_model.pt'):
    model.load_state_dict(torch.load(path))

def plot_rgb_and_gray_images(tensor, name):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3
    rgb_images = tensor[:3]
    gray_images = tensor[3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_images.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("RGB Image")
    axes[0].axis('off')
    axes[1].imshow(gray_images.cpu().numpy(), cmap='gray')
    axes[1].set_title("Gray Image")
    axes[1].axis('off')
    plt.savefig(f'vis/image_{name}.png')
    plt.close(fig)

def aug_data_rotation(obses, picker_states, actions, k=5, visualize=False):
    # obses: shape [N, 4, 168, 168], Sequence of observations with shape [4, 168, 168]
    # picker_states: shape [N, 2], Sequence of picker states
    # action: shape [N, 8], Sequence of actions
    # k: number of randomly rotate sequences
    obses_fn = []
    actions_fn = []
    picker_states_fn = []
    actions_array = actions.cpu().numpy()
    seq_len, channels, height, width = obses.shape
    for i in range(k):
        theta, _, pivot = get_random_image_transform_params(image_size=[height, width])
        transform = get_image_transform(theta, [0., 0.], pivot)
        rot = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
        obs_es = []
        picker_state_s = []
        action_s = []
        for j in range(seq_len):
            if visualize:
                plot_rgb_and_gray_images(obses[j, ...], f'obs_{j}')
            obs = obses[j, ...].numpy().copy()
            picker_state = picker_states[j, ...]
            action = actions_array[j, ...]
            dxy = action[::2].copy()
            dxy1, dxy2 = np.split(dxy, 2)
            # transform action
            rotated_dxy1 = rot.dot(dxy1)
            rotated_dxy1 = np.clip(rotated_dxy1, -1, 1)
            rotated_dxy2 = rot.dot(dxy2)
            rotated_dxy2 = np.clip(rotated_dxy2, -1, 1)
            action_aug = action.copy()
            action_aug[0] = dxy1[0]
            action_aug[2] = dxy1[1]
            action_aug[4] = dxy2[0]
            action_aug[6] = dxy2[1]
            action_s.append(torch.from_numpy(action_aug).to(torch.float32))
            # transform obs
            for k in range(channels):
                obs[k, :, :] = affine_transform(obs[k, :, :], np.linalg.inv(transform), mode='nearest', order=1)
            obs = torch.from_numpy(obs)
            if visualize:
                plot_rgb_and_gray_images(obs, f'obs_aug_{j}')
            obs_es.append(obs)
            picker_state_s.append(picker_state)
        obs_es = torch.stack(obs_es, dim=0)
        picker_state_s = torch.stack(picker_state_s, dim=0)
        action_s = torch.stack(action_s, dim=0)
        obses_fn.append(obs_es)
        picker_states_fn.append(picker_state_s)
        actions_fn.append(action_s)
    obses_fn = torch.stack(obses_fn, dim=0)
    picker_states_fn = torch.stack(picker_states_fn, dim=0)
    actions_fn = torch.stack(actions_fn, dim=0)
    return obses_fn, picker_states_fn, actions_fn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, last_activation='tanh'):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        if last_activation == 'tanh':
            self.last = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last = nn.Sigmoid()
        elif last_activation == 'relu':
            self.last = nn.ReLU()
        elif last_activation == 'GELU':
            self.last = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.last(x)
        return x
        
class RNN_MIMO_MLP(nn.Module):
    """
    Structure: [encoder -> rnn -> mlp -> decoder]
    """
    def __init__(
            self,
            output_dim,
            hidden_dim=256,
            rnn_input_dim=514,
            rnn_hidden_dim=1024,
            rnn_num_layers=2,
            rnn_type="LSTM", # [LSTM, GRU]
            load_obs_pretrained=False,
            finetune_obs_pretrained=False,
            path_pretrained=None
            ):
        """
        output_dim (int): output dimension
        hidden_dim (int): hidden dimension of the MLP
        rnn_input_dim (int): RNN input dimension
        rnn_hidden_dim (int): RNN hidden dimension
        rnn_num_layers (int): number of RNN layers
        rnn_type (str): [LSTM, GRU]
        use_pretrained_0_train: whether to use pretrained model for 0 training
        path_pretrained: path to pretrained model
        """
        super(RNN_MIMO_MLP, self).__init__()
        self.load_obs_pretrained = load_obs_pretrained
        self.finetune_obs_pretrained = finetune_obs_pretrained
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.load_obs_pretrained:
            assert path_pretrained is not None, "Path to pretrained model is None"
            self.model.load_state_dict(torch.load(path_pretrained))
        self.model.fc = nn.Identity()
        if self.finetune_obs_pretrained is False:
            for param in self.model.parameters():
                param.requires_grad = False
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.mlp = MLP(input_dim=rnn_hidden_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        
    def forward(self, obs, picker_state, rnn_hidden_state=None):
        """
        Training:
            obs: [B, T, C, H, W]
            picker_state: [B, T, 2]
        Eavaluation:
            obs: [B, C, H, W]
            picker_state: [B, 2]
        Retuns:
            outputs: outputs of the per step net
            rnn_state: return rnn state if return_state is True
        """
        if obs.ndim == 5 and picker_state.ndim == 3:
            # Training mode
            batch, length, channel, height, width = obs.size()
            if self.finetune_obs_pretrained is False:
                with torch.no_grad():
                    inputs = self.model(obs.view(-1, channel, height, width))
            else:
                inputs = self.model(obs.view(-1, channel, height, width))
            inputs = inputs.view(batch, length, -1)
            inputs = torch.cat((inputs, picker_state), dim=-1)
            assert inputs.dim() == 3, "Input dimension should be 3"
            outputs, _ = self.rnn(inputs)
            outputs = self.mlp(outputs)
            return outputs
        elif obs.ndim == 4 and picker_state.ndim == 2:
            assert rnn_hidden_state is not None, "RNN hidden state is None"
            # Evaluation mode
            obs = obs.unsqueeze(1)
            picker_state = picker_state.unsqueeze(1)
            batch, _, channel, height, width = obs.size()
            with torch.no_grad():
                inputs = self.model(obs.view(-1, channel, height, width))
                inputs = inputs.view(batch, 1, -1)
                inputs = torch.cat((inputs, picker_state), dim=-1)
                assert inputs.dim() == 3, "Input dimension should be 3"
                try:
                    outputs, rnn_hidden_state = self.rnn(inputs, rnn_hidden_state)
                except:
                    outputs, rnn_hidden_state = self.rnn(inputs, (rnn_hidden_state[0].to(obs.device), rnn_hidden_state[1].to(obs.device)))
                outputs = self.mlp(outputs)
                return outputs, rnn_hidden_state
        else:
            raise ValueError("Input dimension is not correct")
        
    def reset(self):
        # Initialize the hidden state
        return (torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim),
                torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim))

class BC_model(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim=256,
                 load_obs_pretrained=False,
                 finetune_obs_pretrained=False, 
                 path_pretrained=None):
        super(BC_model, self).__init__()
        self.load_obs_pretrained = load_obs_pretrained
        self.finetune_obs_pretrained = finetune_obs_pretrained
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.load_obs_pretrained:
            assert path_pretrained is not None, "Path to pretrained model is None"
            self.model.load_state_dict(torch.load(path_pretrained, map_location=torch.device('cpu')))
        self.model.fc = nn.Identity()
        if self.finetune_obs_pretrained is False:
            for param in self.model.parameters():
                param.requires_grad = False
        self.mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    def forward(self, obs, picker_state):
        if self.finetune_obs_pretrained is False:
            with torch.no_grad():
                repr = self.model(obs)
        else:
            repr = self.model(obs)
        repr = torch.cat((repr, picker_state), dim=1)
        out = self.mlp(repr)
        return out

class BeT_model(nn.Module):
    def __init__(self,
                 input_dim,
                 act_dim,
                 n_clusters=64,
                 k_means_fit_steps=1000,
                 gpt_block_size=30,
                 gpt_n_layer=3,
                 gpt_n_head=4,
                 gpt_n_embd=256,
                 load_obs_pretrained=False,
                 finetune_obs_pretrained=False,
                 path_pretrained=None):
        super(BeT_model, self).__init__()
        self.load_obs_pretrained = load_obs_pretrained
        self.finetune_obs_pretrained = finetune_obs_pretrained
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.load_obs_pretrained:
            assert path_pretrained is not None, "Path to pretrained model is None"
            self.model.load_state_dict(torch.load(path_pretrained))
        self.model.fc = nn.Identity()
        if self.finetune_obs_pretrained is False:
            for param in self.model.parameters():
                param.requires_grad = False

        self.gpt = GPT(GPTConfig(block_size=gpt_block_size,
                                 input_dim=input_dim,
                                 n_layer=gpt_n_layer,
                                 n_head=gpt_n_head,
                                 n_embd=gpt_n_embd))
        
        self.cbet = BehaviorTransformer(obs_dim=input_dim,
                                        act_dim=act_dim,
                                        goal_dim=0,
                                        gpt_model=self.gpt,
                                        n_clusters=n_clusters,
                                        kmeans_fit_steps=k_means_fit_steps)

    def forward(self, obs, picker_state, action=None):
        batch, length, channel, height, width = obs.size()
        if self.finetune_obs_pretrained is False:
            with torch.no_grad():
                inputs = self.model(obs.view(-1, channel, height, width))
        else:
            inputs = self.model(obs.view(-1, channel, height, width))
        inputs = inputs.view(batch, length, -1)
        inputs = torch.cat((inputs, picker_state), dim=2)
        assert inputs.dim() == 3, "Input dimension should be 3"
        goals = torch.zeros(batch, length, 0).to(obs.device)
        if action is None:
            out = self.cbet(inputs, goals, action)[0]
            return out[:, -1, :]
        else:
            return self.cbet(inputs, goals, action)[1]

class NPYDataset(Dataset):
    def __init__(self, 
                 npy_files, 
                 columns=['NPY_Path'], 
                 play=False, 
                 representation_learning=False, 
                 transform_color=False, 
                 transform_rot=False,
                 num_play_obs=120,
                 num_aug_rotation=8):
        self.data = pd.read_csv(npy_files, usecols=columns).values
        self.play = play # whether use play data for representation learning
        self.representation_learning = representation_learning # whether use representation learning
        self.transform_color = transform_color
        self.transform_rot = transform_rot
        self.num_play_obs = num_play_obs
        self.num_aug_rotation = num_aug_rotation
        if self.transform_color:
            self.transform = T.Compose([
            RandomChangeBackgroundRGBD(p=0.5),
            RandomColorJitterRGBD(p=0.5),
            RandomGrayScaleRGBD(p=0.3),
        ])
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.play:
            npy_file = self.data[idx][0]
            frame = np.load(npy_file, allow_pickle=True)
            obses = [frame[i][0] for i in range(len(frame))]
            obses = torch.stack(obses, dim=0)
            return obses
        else:
            _, final_step_str, npy_file = self.data[idx]
            frame = np.load(npy_file, allow_pickle=True)
            final_step_str = final_step_str.replace('[', '').replace(']', '').split(',')
            final_step = [int(i) for i in final_step_str]
            obses = []
            picker_states = []
            actions = []
            steps = []
            for i in range(1, len(frame) + 1):
                obs = frame[i-1][0]
                obses.append(obs.squeeze(0))
                picker_state = torch.from_numpy(frame[i-1][5]).to(torch.float32)
                picker_states.append(picker_state)
                action = torch.from_numpy(frame[i-1][1]).to(torch.float32)
                actions.append(action)
                step = final_step[-1] - i + 1
                steps.append(step)
            obses = torch.stack(obses, dim=0)
            picker_states = torch.stack(picker_states, dim=0)
            actions = torch.stack(actions, dim=0)
            steps = torch.from_numpy(np.array(steps)).to(torch.float32)
            goal = frame[final_step[-1]-1][3]
            goal_picker_state = torch.from_numpy(frame[final_step[-1]-1][6]).to(torch.float32)
            if self.representation_learning:
                if len(obses) > self.num_play_obs:
                    random_idx = np.random.randint(0, len(obses)-self.num_play_obs)
                    obses = obses[random_idx:random_idx+self.num_play_obs]
                return obses
            else:
                if self.transform_color:
                    obses = self.transform(obses)
                    goal = self.transform(goal)
                if self.transform_rot:
                    obses, picker_states, actions = aug_data_rotation(obses, 
                                                                      picker_states, 
                                                                      actions, 
                                                                      self.num_aug_rotation, 
                                                                      False)
                return tuple([obses, picker_states, actions, goal, goal_picker_state, steps])

class BeT_Dataset(Dataset):
    def __init__(self, 
                 npy_file, 
                 columns=['Length', 'NPY_Path'],
                 transform_color=False,
                 transform_rot=False,
                 num_aug_rotation=8,
                 window_size=10):
        # load file npy
        self.dataset = pd.read_csv(npy_file, usecols=columns).values
        # take the number rows of dataset
        self.transform_color = transform_color
        self.transform_rot = transform_rot
        self.num_aug_rotation = num_aug_rotation
        self.window_size = window_size
        self.slices = []
        for i in range(len(self.dataset)):
            T = self.dataset[i][0] - 1
            for j in range(T - self.window_size):
                self.slices.append((i, j, j + self.window_size))
        if self.transform_color:
            self.transform = T.Compose([
            RandomChangeBackgroundRGBD(p=0.5),
            RandomColorJitterRGBD(p=0.5),
            RandomGrayScaleRGBD(p=0.3),
        ])
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        data = np.load(self.dataset[i][1], allow_pickle=True)
        obs = []
        picker_state = []
        action = []
        data = data[start:end]
        for j in range(len(data)):
            obs.append(data[j][0].squeeze(0))
            picker_state.append(torch.from_numpy(data[j][5]).to(torch.float32))
            action.append(torch.from_numpy(data[j][1]).to(torch.float32))
        obs = torch.stack(obs, dim=0)
        picker_state = torch.stack(picker_state, dim=0)
        action = torch.stack(action, dim=0)
        if self.transform_color:
            obs = self.transform(obs)
        if self.transform_rot:
            obs, picker_state, action = aug_data_rotation(obs, 
                                                          picker_state, 
                                                          action, 
                                                          self.num_aug_rotation, 
                                                          False)
        assert obs.shape[0] == self.window_size
        assert picker_state.shape[0] == self.window_size
        assert action.shape[0] == self.window_size
        return tuple([obs, picker_state, action])
    
def discount_cumsum(T, start, end, gamma=0.99):
    discount_cumsum = np.zeros(T)
    discount_cumsum[-1] = 1
    for t in reversed(range(T-1)):
        discount_cumsum[t] += gamma * discount_cumsum[t+1]
    discount_cumsum = discount_cumsum[start:end]
    return discount_cumsum

class DT_Dataset(BeT_Dataset):
    def __init__(self, npy_file, columns=['Length', 'NPY_Path'], window_size=10):
        # load file npy
        self.dataset = pd.read_csv(npy_file, usecols=columns).values
        # take the number rows of dataset
        self.window_size = window_size
        self.slices = []
        for i in range(len(self.dataset)):
            T = self.dataset[i][0] - 1
            for j in range(T - self.window_size):
                self.slices.append((i, j, j + self.window_size, T+1))

    def __getitem__(self, idx):
        i, start, end, T = self.slices[idx]
        data = np.load(self.dataset[i][1], allow_pickle=True)
        obs = []
        picker_state = []
        action = []
        data = data[start:end]
        for j in range(len(data)):
            obs.append(data[j][0].to(torch.float32))
            picker_state.append(torch.from_numpy(data[j][5]).to(torch.float32))
            action.append(torch.from_numpy(data[j][1]).to(torch.float32))
        obs = torch.stack(obs, dim=0).squeeze(1)
        picker_state = torch.stack(picker_state, dim=0)
        action = torch.stack(action, dim=0)
        rtg = discount_cumsum(T, start, end)
        timestep = torch.from_numpy(np.arange(start, end)).to(torch.float32)
        assert rtg.shape[0] == self.window_size
        assert obs.shape[0] == self.window_size
        assert picker_state.shape[0] == self.window_size
        assert action.shape[0] == self.window_size
        return tuple([obs, picker_state, action, rtg, timestep])

def train_representation_learning(dataloader_play, dataloader_demo, device, img_size=224, num_epochs=100):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.to(device)
    losses = []
    learner = BYOL(
        model,
        image_size=img_size,
        hidden_layer=-2,
    )
    learner = learner.to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    scaler = GradScaler()
    for _ in range (num_epochs):
        l = 0
        # for idx, data in tqdm(enumerate(dataloader_play)):
        for data_play, data_demo in tqdm(zip(dataloader_play, dataloader_demo)):
            obs_play = data_play
            obs_play = obs_play.to(device).squeeze(0).squeeze(1)
            obs_demo= data_demo
            obs_demo = obs_demo.to(device).squeeze(0)
            # Concatenate the observations
            obs = torch.cat((obs_play, obs_demo), dim=0)
            opt.zero_grad()
            with autocast():
                loss = learner(obs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            learner.update_moving_average()
            l += loss.item()
        losses.append(l/len(dataloader_play))
        if (_ + 1) % 5 == 0:   
            print('[INFO] Epoch: {}, Loss: {}'.format(_+1, l/len(dataloader_play)))

    # Save the model
    torch.save(model.state_dict(), './models/pretrained_observation_model.pt')
    print(f'[INFO] Saved model to ./models/pretrained_observation_model.pt')
    # Plot the losses
    plt.plot(losses)
    plt.title('Representation Learning Losses (BYOL)')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('./vis/representation_losses.png')
    plt.close()

def infer_representation_learning(dataloader_demo, device):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    load_model(model, path='./models/pretrained_observation_model.pt')
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    reprs = []
    actions = []
    goals = []
    for data_demo in tqdm(dataloader_demo):
        obs = data_demo[0].to(device).squeeze(0)
        obs = obs.to(torch.float32) / 255.
        picker_state = data_demo[1].to(device).squeeze(0)
        action = data_demo[2].to(device).squeeze(0)
        goal = data_demo[3].to(device).squeeze(0)
        goal = goal.to(torch.float32) / 255.
        goal_picker_state = data_demo[4].to(device)
        with torch.no_grad():
            repr = model(obs)
            goal = model(goal)
        repr = torch.cat((repr, picker_state), dim=1)
        goal = torch.cat((goal, goal_picker_state), dim=1)
        goal = goal.repeat(repr.shape[0], 1)
        reprs.append(repr)
        actions.append(action)
        goals.append(goal)
    reprs = torch.cat(reprs, dim=0)
    actions = torch.cat(actions, dim=0)
    goals = torch.cat(goals, dim=0)
    torch.save(reprs, './models/reprs.pt')
    print(f'[INFO] Saved representations to ./models/reprs.pt')
    torch.save(actions, './models/actions.pt')
    print(f'[INFO] Saved actions to ./models/actions.pt')
    torch.save(goals, './models/goals.pt')
    print(f'[INFO] Saved goals to ./models/goals.pt')

def train_BC(model, train_loader, test_loader, device, type_model, transform_color, transform_rot, load_obs_pretrained, finetune_obs_pretrained, num_epochs=100):
    name = type_model
    if transform_color:
        name += '_aug_color'
    if transform_rot:
        name += '_aug_rot'
    if load_obs_pretrained:
        name += '_load_obs_pretrained'
    if finetune_obs_pretrained:
        name += '_finetune_obs_pretrained'
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_loss = []
    test_loss = []
    best_test_loss = np.inf
    scaler = GradScaler()
    for _ in range(num_epochs):
        model.train()
        l = 0
        for data in tqdm(train_loader):
            obs, picker_state, action = data[0].to(device), data[1].to(device), data[2].to(device)
            obs = obs.to(torch.float32) / 255.
            if transform_rot:
                obs = obs.squeeze(0)
                picker_state = picker_state.squeeze(0)
                action = action.squeeze(0)
            if type_model == 'BC':
                _, c, h, w = obs.shape
                obs = obs.view(-1, c, h, w)
                picker_state = picker_state.view(-1, 2)
                action = action.view(-1, 8)
                with autocast():
                    pred = model(obs, picker_state)
                    loss = nn.MSELoss()(pred, action)
            elif type_model == 'BC_RNN':
                with autocast():
                    pred = model(obs, picker_state)
                    loss = nn.MSELoss()(pred, action)
            elif type_model == 'BeT':
                with autocast():
                    loss = model(obs, picker_state, action)
            else:
                raise ValueError("Model name is not correct")
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 25)
            scaler.step(opt)
            scaler.update()
            l += loss.item()
        train_loss.append(l/len(train_loader))
        
        if (_ + 1) % 5 == 0:
            print('[INFO] Epoch: {}, Train Loss: {}'.format(_+1, l/len(train_loader)))
            # Evaluate the model
            model.eval()
            l_t = 0.
            for data_test in tqdm(test_loader):
                obs, picker_state, action = data_test[0].to(device), data_test[1].to(device), data_test[2].to(device)
                obs = obs.to(torch.float32) / 255.
                if transform_rot:
                    obs = obs.squeeze(0)
                    picker_state = picker_state.squeeze(0)
                    action = action.squeeze(0)
                if type_model == 'BC':
                    _, c, h, w = obs.shape
                    obs = obs.view(-1, c, h, w)
                    picker_state = picker_state.view(-1, 2)
                    action = action.view(-1, 8)
                    with torch.no_grad():
                        pred = model(obs, picker_state)
                        loss = nn.MSELoss()(pred, action)
                elif type_model == 'BC_RNN':
                    with torch.no_grad():
                        pred = model(obs, picker_state)
                        loss = nn.MSELoss()(pred, action)
                elif type_model == 'BeT':
                    with torch.no_grad():
                        loss = model(obs, picker_state, action)
                l_t += loss.item()
            test_loss.append(l_t/len(test_loader))
            print('[INFO] Epoch: {}, Test Loss: {}'.format(_+1, l_t/len(test_loader)))
            if l_t/len(test_loader) < best_test_loss:
                best_test_loss = l_t/len(test_loader)
                torch.save(model.state_dict(), f'./models/{name}.pt')
                print('[INFO] Model saved')

    # # Plot the Train & Test Loss
    plt.figure()
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title(f'{name} Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend(['Train', 'Test'])
    plt.savefig(f'./vis/{name}_losses.png')
    plt.close()
    
def train_dynamical_step(model, dynamical_model, expert_dataloader, device, img_size=224, num_epochs=100, k=32):
    model.eval()
    dynamical_model.train()
    dynamical_opt = torch.optim.Adam(dynamical_model.parameters(), lr=3e-4)
    dynamical_losses = []
    best_loss = np.inf
    for _ in range(num_epochs):
        l = 0
        for traj in expert_dataloader:
            obs = torch.stack([t[0] for t in traj], dim=0).view(-1, 4, img_size, img_size).to(device)
            picker_state = torch.stack([t[1] for t in traj], dim=0).view(-1, 2).to(device)
            goal_obs = torch.stack([t[3] for t in traj], dim=0).view(-1, 4, img_size, img_size).to(device)
            goal_picker_state = torch.stack([t[4] for t in traj], dim=0).view(-1, 2).to(device)
            step = torch.stack([t[5] for t in traj], dim=0).view(-1, 1).to(device)
            step /= 100.0
            with torch.no_grad():
                repr = model(obs)
                goal_repr = model(goal_obs)
            # Concatenate the picker state
            repr = torch.cat((repr, picker_state), dim=1)
            goal_repr = torch.cat((goal_repr, goal_picker_state), dim=1)

            # Train the dynamical model
            pred = dynamical_model(repr, goal_repr)
            loss = nn.MSELoss()(pred, step)
            dynamical_opt.zero_grad()
            loss.backward()
            dynamical_opt.step()
            l += loss.item()

        dynamical_losses.append(l)
        if (_ + 1) % 10 == 0:
            print('[INFO] Epoch: {}, Loss: {}'.format(_+1, l))
        if l < best_loss:
            best_loss = l
            torch.save(dynamical_model.state_dict(), 'dynamical_model.pt')

    # Plot the losses
    plt.figure()
    plt.plot(dynamical_losses)
    plt.title('Dynamical Model Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('dynamical_losses.png')
    plt.close()

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--type_model', type=str, default='BC_RNN', choices=['BeT', 'BC', 'BC_RNN', 'Representation Learning'])
    argp.add_argument('--device', type=str, default='cuda')
    argp.add_argument('--batch_size', type=int, default=1)
    argp.add_argument('--path_to_play_data', type=str, default='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/play.csv')
    argp.add_argument('--path_to_demo_data', type=str, default='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv')
    argp.add_argument('--transform_color', type=bool, default=True, help='Change backgound and color jitter of the observations')
    argp.add_argument('--transform_rot', type=bool, default=True, help='Rotate the observations')
    argp.add_argument('--num_aug_rotation', type=int, default=6, help='Number of rotations')
    argp.add_argument('--load_obs_pretrained', type=bool, default=False, help='whether to load the pretrained observation model')
    argp.add_argument('--finetune_obs_pretrained', type=bool, default=True, help='whether to finetune the pretrained observation model')
    argp.add_argument('--train_representation_learning', type=bool, default=True, help='whether to train the representation learning')
    argp.add_argument('--num_epochs', type=int, default=100)
    args = argp.parse_args()

    device = torch.device(args.device)
    if args.type_model == 'Representation Learning':
        # Load the dataset
        dataset_play = NPYDataset(npy_files=args.path_to_play_data, 
                                  columns=['NPY_Path'], 
                                  play=True)
        dataloader_play = DataLoader(dataset_play, batch_size=args.batch_size, shuffle=True, num_workers=16)
        if args.train_representation_learning:
            dataset_demo = NPYDataset(npy_files=args.path_to_demo_data, 
                                      columns=['Length', 'Final_step', 'NPY_Path'], 
                                      play=False, 
                                      representation_learning=True)
            dataloader_demo = DataLoader(dataset_demo, batch_size=args.batch_size, shuffle=True, num_workers=16)
            # Train the representation learning
            train_representation_learning(dataloader_play,
                                          dataloader_demo, 
                                          device, 
                                          img_size=168, 
                                          num_epochs=args.num_epochs)
        else:
            dataset_demo = NPYDataset(npy_files=args.path_to_demo_data,
                                      columns=['Length', 'Final_step', 'NPY_Path'],
                                      play=False,
                                      representation_learning=False,
                                      )
            dataloader_demo = DataLoader(dataset_demo, batch_size=args.batch_size, shuffle=True, num_workers=16)
            infer_representation_learning(dataloader_demo, device)
        exit()
    elif args.type_model == 'BC' or args.type_model == 'BC_RNN':
        # Load dataset
        dataset_demo = NPYDataset(npy_files=args.path_to_demo_data, 
                                  columns=['Length', 'Final_step', 'NPY_Path'], 
                                  transform_color=args.transform_color,
                                  transform_rot=args.transform_rot,
                                  num_aug_rotation=args.num_aug_rotation)
        # Define the model
        if args.type_model == 'BC':
            model = BC_model(input_dim=514,
                            output_dim=8,
                            load_obs_pretrained=args.load_obs_pretrained,
                            finetune_obs_pretrained=args.finetune_obs_pretrained).to(device)
        else:
            model = RNN_MIMO_MLP(output_dim=8,
                                 load_obs_pretrained=args.load_obs_pretrained,
                                 finetune_obs_pretrained=args.finetune_obs_pretrained).to(device)
    elif args.type_model == 'BeT':
        # Load dataset
        dataset_demo = BeT_Dataset(npy_file=args.path_to_demo_data, 
                                   columns=['Length', 'Final_step', 'NPY_Path'],
                                   transform_color=args.transform_color,
                                   transform_rot=args.transform_rot,
                                   num_aug_rotation=args.num_aug_rotation,
                                   window_size=10)
        model = BeT_model(input_dim=514,
                          act_dim=8,
                          k_means_fit_steps=500,
                          load_obs_pretrained=args.load_obs_pretrained,
                          finetune_obs_pretrained=args.finetune_obs_pretrained).to(device)
                    
    # Split to train and test with 80:20
    train_size = int(0.8 * len(dataset_demo))
    test_size = len(dataset_demo) - train_size
    train_dataset, test_dataset = random_split(dataset_demo, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    # Train model
    train_BC(model,
             train_loader,
             test_loader,
             device,
             args.type_model,
             args.transform_color,
             args.transform_rot,
             args.load_obs_pretrained,
             args.finetune_obs_pretrained,
             args.num_epochs)
    
    # Define the dynamical model
    # dynamical_model = Dynamical_Model(input_dim=514, output_dim=1, hidden_dim=256).to(device)
    # train_dynamical_step(model, dynamical_model, expert_dataloader, device, img_size=168, num_epochs=100)
    # dynamical_model = load_model(dynamical_model, path='/home/hnguyen/cloth_smoothing/equiRL/dynamical_model.pt', representation=False)
