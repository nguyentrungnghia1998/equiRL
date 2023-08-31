import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from byol import BYOL
from tqdm import tqdm
from behavior_transformer import BehaviorTransformer, GPT, GPTConfig

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, last_activation='tanh'):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x
    
class Dynamical_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, last_activation='sigmoid'):
        super(Dynamical_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        if last_activation == 'tanh':
            self.last = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        # concatenate
        x = torch.cat((x1, x2), dim=1)
        x = self.fc2(x)
        x = self.relu(x)
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
            hidden_dim,
            rnn_input_dim,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type="LSTM", # [LSTM, GRU]
            use_pretrained_0_train=False,
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
        assert path_pretrained is not None, "Path to pretrained model is None"
        self.use_pretrained_0_train = use_pretrained_0_train
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.load_state_dict(torch.load(path_pretrained, map_location=torch.device('cpu')))
        self.model.fc = nn.Identity()
        if self.use_pretrained_0_train:
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
            if self.use_pretrained_0_train:
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
    def __init__(self, input_dim, output_dim, hidden_dim=256, use_pretrained_0_train=False, path_pretrained=None):
        super(BC_model, self).__init__()
        assert path_pretrained is not None, "Path to pretrained model is None"
        self.use_pretrained_0_train = use_pretrained_0_train
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.load_state_dict(torch.load(path_pretrained, map_location=torch.device('cpu')))
        self.model.fc = nn.Identity()
        if self.use_pretrained_0_train:
            for param in self.model.parameters():
                param.requires_grad = False
        self.mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    def forward(self, obs, picker_state):
        if self.use_pretrained_0_train:
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
                 gpt_block_size=200,
                 gpt_n_layer=6,
                 gpt_n_head=4,
                 gpt_n_embd=256,
                 use_pretrained_0_train=False,
                 path_pretrained=None):
        super(BeT_model, self).__init__()
        assert path_pretrained is not None, "Path to pretrained model is None"
        self.use_pretrained_0_train = use_pretrained_0_train
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.load_state_dict(torch.load(path_pretrained, map_location=torch.device('cpu')))
        self.model.fc = nn.Identity()
        if use_pretrained_0_train:
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
        if self.use_pretrained_0_train:
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
            print(out[:, -1, :])
            return out[:, -1, :]
        else:
            return self.cbet(inputs, goals, action)[1]
                       
def load_model(model, path='/home/hnguyen/cloth_smoothing/equiRL/learn_repr.pt'):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Define a custom dataset to load the NPY files
class NPYDataset(Dataset):
    def __init__(self, npy_files, columns=['NPY_Path'], play=True, representation_learning=False, rnn=False):
        self.data = pd.read_csv(npy_files, usecols=columns).values
        self.rnn = rnn
        self.play = play # whether use play data for representation learning
        self.representation_learning = representation_learning
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.play:
            npy_file = self.data[idx][0]
            frame = np.load(npy_file, allow_pickle=True)
            obses = []
            next_obses = []
            for i in range(len(frame)):
                obs = frame[i][0]
                obses.append(obs.squeeze(0))
            obses = torch.stack(obses, dim=0)
            return obses
        else:
            length, final_step_str, npy_file = self.data[idx]
            frame = np.load(npy_file, allow_pickle=True)
            final_step_str = final_step_str.replace('[', '').replace(']', '').split(',')
            final_step = [int(i) for i in final_step_str]
            obses = []
            actions = []
            next_obses = []
            picker_states = []
            next_picker_states = []
            goals = []
            goal_picker_states = []
            steps = []
            j = 0
            for i in range(1, len(frame) + 1):
                obs = frame[i-1][0]
                picker_state = torch.from_numpy(frame[i-1][5]).to(torch.float32)
                action = torch.from_numpy(frame[i-1][1]).to(torch.float32)
                next_obs = frame[i-1][3].to(torch.float32) / 255.0
                next_picker_state = torch.from_numpy(frame[i-1][6]).to(torch.float32)
                goal = frame[final_step[j]-1][3].to(torch.float32) / 255.0
                goal_picker_state = torch.from_numpy(frame[final_step[j]-1][6]).to(torch.float32)
                step = final_step[j] - i + 1
                obses.append(obs.squeeze(0))
                actions.append(action)
                next_obses.append(next_obs.squeeze(0))
                picker_states.append(picker_state)
                next_picker_states.append(next_picker_state)
                goals.append(goal.squeeze(0))
                goal_picker_states.append(goal_picker_state)
                steps.append(step)
                if i == final_step[j]:
                    j += 1
                # if self.rnn and j == 3:
                #     break
            obses = torch.stack(obses, dim=0)
            picker_states = torch.stack(picker_states, dim=0)
            actions = torch.stack(actions, dim=0)
            next_obses = torch.stack(next_obses, dim=0)
            next_picker_states = torch.stack(next_picker_states, dim=0)
            goals = torch.stack(goals, dim=0)
            goal_picker_states = torch.stack(goal_picker_states, dim=0)
            steps = torch.from_numpy(np.array(steps)).to(torch.float32)
            if self.representation_learning:
                if len(obses) > 120:
                    random_idx = np.random.randint(0, len(obses)-120)
                    obses = obses[random_idx:random_idx+120]
                return obses
            else:
                obses = obses.to(torch.float32) / 255.0
                return tuple([obses, actions, next_obses, goals, picker_states, next_picker_states, goal_picker_states, steps])

class BeT_Dataset(Dataset):
    def __init__(self, 
                 npy_file, 
                 columns=['Length', 'NPY_Path'], 
                 window_size=10):
        # load file npy
        self.dataset = pd.read_csv(npy_file, usecols=columns).values
        # take the number rows of dataset
        self.window_size = window_size
        self.slices = []
        for i in range(len(self.dataset)):
            T = self.dataset[i][0] - 1
            for j in range(T - self.window_size):
                self.slices.append((i, j, j + self.window_size))
    
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
            obs.append(data[j][0].to(torch.float32) / 255.0)
            picker_state.append(torch.from_numpy(data[j][5]).to(torch.float32))
            action.append(torch.from_numpy(data[j][1]).to(torch.float32))
        obs = torch.stack(obs, dim=0).squeeze(1)
        picker_state = torch.stack(picker_state, dim=0)
        action = torch.stack(action, dim=0)
        assert obs.shape[0] == self.window_size
        assert picker_state.shape[0] == self.window_size
        assert action.shape[0] == self.window_size
        return tuple([obs, picker_state, action])
    
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
            obs_play = obs_play.to(device).squeeze(0)
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
    torch.save(model.state_dict(), './model/pretrained_observation_model.pt')
    # Plot the losses
    plt.plot(losses)
    plt.title('Representation Learning Losses (BYOL)')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('./model/representation_losses.png')
    plt.close()

def train_BC(model, dataloader_demo, device, num_epochs=100, use_pretrained_0_train=False, name_model="BC"):
    model.train()
    bc_opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    bc_losses = []
    best_loss = np.inf
    scaler = GradScaler()
    for _ in range(num_epochs):
        l = 0
        for i, data in tqdm(enumerate(dataloader_demo)):
            if name_model == "BC":
                obs = data[0].to(device).squeeze(0)
                picker_state = data[4].to(device).squeeze(0)
                action = data[1].to(device).squeeze(0)
            elif name_model == "BC_RNN":
                obs = data[0].to(device)
                picker_state = data[4].to(device)
                action = data[1].to(device)
            elif name_model == "BeT":
                obs = data[0].to(device)
                picker_state = data[1].to(device)
                action = data[2].to(device)
            else:
                raise ValueError("Model name is not correct")
            bc_opt.zero_grad()
            with autocast():
                if name_model == "BeT":
                    loss = model(obs, picker_state, action)
                else:
                    pred = model(obs, picker_state)
                    loss = nn.MSELoss()(pred, action)
            if use_pretrained_0_train:
                for param in model.model.parameters():
                    assert param.requires_grad == False
            else:
                for param in model.model.parameters():
                    assert param.requires_grad == True
            scaler.scale(loss).backward()
            scaler.unscale_(bc_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(bc_opt)
            scaler.update()
            l += loss.item()
        bc_losses.append(l/len(dataloader_demo))
        if (_ + 1) % 5 == 0:
            print('[INFO] Epoch: {}, Loss: {}'.format(_+1, l/len(dataloader_demo)))
        if l/len(dataloader_demo) < best_loss:
            best_loss = l/len(dataloader_demo)
            if use_pretrained_0_train:
                torch.save(model.state_dict(), f'./model/{name_model}_0_finetune_pretrained.pt')
            else:
                torch.save(model.state_dict(), f'./model/{name_model}_1.pt')
    # Plot the losses
    plt.figure()
    plt.plot(bc_losses)
    plt.title(f'{name_model} Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    if use_pretrained_0_train:
        plt.savefig(f'./model/{name_model}_0_finetune_pretrained.png')
    else:
        plt.savefig(f'./model/{name_model}_1.png')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the dataset
    batch_size = 1
    dataset_play = NPYDataset(npy_files='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/play.csv', columns=['NPY_Path'], play=True)
    dataset_demo = NPYDataset(npy_files='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv', columns=['Length', 'Final_step', 'NPY_Path'], play=False, representation_learning=True)
    dataloader_play = DataLoader(dataset_play, batch_size=1, shuffle=True, num_workers=4)
    dataloader_demo = DataLoader(dataset_demo, batch_size=1, shuffle=True, num_workers=4)
    # Train the representation learning
    train_representation_learning(dataloader_play, dataloader_demo, device, img_size=168, num_epochs=50)
    exit()
    
    # # Define the BC model
    # dataset_demo = NPYDataset(npy_files='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv', columns=['Length', 'Final_step', 'NPY_Path'], play=False, representation_learning=False)
    # dataloader_demo = DataLoader(dataset_demo, batch_size=1, shuffle=False, num_workers=4)
    # Train BC model use pretrained model but not finetune 
    # bc_model_0_finetune_pretrained = BC_model(input_dim=514,
    #                                        output_dim=8,
    #                                        hidden_dim=256,
    #                                        use_pretrained_0_train=True,
    #                                        path_pretrained='/home/hnguyen/cloth_smoothing/equiRL/model/pretrained_observation_model.pt').to(device)
    # train_BC(model=bc_model_0_finetune_pretrained,
    #          dataloader_demo=dataloader_demo,
    #          device=device,
    #          num_epochs=100,
    #          use_pretrained_0_train=True, 
    #          name_model="BC")
    # Train BC model use pretrained model and finetune
    # bc_model_finetune_pretrained = BC_model(input_dim=514,
    #                             output_dim=8,
    #                             hidden_dim=256,
    #                             use_pretrained_0_train=False,
    #                             path_pretrained='/home/hnguyen/cloth_smoothing/equiRL/model/pretrained_observation_model.pt').to(device)
    # train_BC(model=bc_model_finetune_pretrained,
    #          dataloader_demo=dataloader_demo,
    #          device=device,
    #          num_epochs=100,
    #          use_pretrained_0_train=False,
    #          use_rnn=False)
    # Define BC RNN model
    dataset_demo_rnn = NPYDataset(npy_files='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv', columns=['Length', 'Final_step', 'NPY_Path'], play=False, representation_learning=False, rnn=True)    
    dataloader_rnn = DataLoader(dataset_demo_rnn, batch_size=1, shuffle=False, num_workers=16)

    # bc_rnn_model_0_finetune_pretrained = RNN_MIMO_MLP(output_dim=8,
    #                                                   hidden_dim=256,
    #                                                   rnn_input_dim=514,
    #                                                   rnn_hidden_dim=1024,
    #                                                   rnn_num_layers=2,
    #                                                   rnn_type="LSTM",
    #                                                   use_pretrained_0_train=True,
    #                                                   path_pretrained='/home/hnguyen/cloth_smoothing/equiRL/model/pretrained_observation_model.pt').to(device)
    # train_BC(model=bc_rnn_model_0_finetune_pretrained,
    #          dataloader_demo=dataloader_rnn,
    #          device=device,
    #          num_epochs=50,
    #          use_pretrained_0_train=True,
    #          use_rnn=True)
    bc_rnn_model_finetune_pretrained = RNN_MIMO_MLP(output_dim=8,
                                                    hidden_dim=256,
                                                    rnn_input_dim=514,
                                                    rnn_hidden_dim=1024,
                                                    rnn_num_layers=2,
                                                    rnn_type="LSTM",
                                                    use_pretrained_0_train=False,
                                                    path_pretrained='/home/hnguyen/cloth_smoothing/equiRL/model/pretrained_observation_model.pt').to(device)
    
    train_BC(model=bc_rnn_model_finetune_pretrained,
             dataloader_demo=dataloader_rnn,
             device=device,
             num_epochs=100,
             use_pretrained_0_train=False,
             name_model="BC_RNN")
    exit()
    # dataset_bet = BeT_Dataset(npy_file='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv', )
    # dataloader_bet = DataLoader(dataset_bet, batch_size=100, shuffle=True, num_workers=16)
    
    # bet_model = BeT_model(input_dim=514,
    #                       act_dim=8,
    #                       n_clusters=64,
    #                       k_means_fit_steps=500,
    #                       use_pretrained_0_train=False,
    #                       path_pretrained='/home/hnguyen/cloth_smoothing/equiRL/model/BC_0_finetune_pretrained.pt').to(device)
    # train_BC(model=bet_model,
    #          dataloader_demo=dataloader_bet,
    #          device=device,
    #          num_epochs=30,
    #          use_pretrained_0_train=False,
    #          name_model="BeT")
    # exit()
    # Define the dynamical model
    # dynamical_model = Dynamical_Model(input_dim=514, output_dim=1, hidden_dim=256).to(device)
    # train_dynamical_step(model, dynamical_model, expert_dataloader, device, img_size=168, num_epochs=100)
    # dynamical_model = load_model(dynamical_model, path='/home/hnguyen/cloth_smoothing/equiRL/dynamical_model.pt', representation=False)


    # Save the representation and action
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)    
    model = load_model(model, path='/home/hnguyen/cloth_smoothing/equiRL/model/pretrained_observation_model.pt')
    model.fc = nn.Identity()
    model.to(device)
    reprs = []
    actions = []
    goals = []
    for traj in tqdm(dataloader_demo):
        obs = traj[0].to(device).squeeze(0)
        picker_state = traj[4].to(device).squeeze(0)
        goal = traj[3].to(device).squeeze(0)
        goal_picker_state = traj[6].to(device).squeeze(0)
        action = traj[1].to(device).squeeze(0)
        with torch.no_grad():
            repr = model(obs)
            goal_repr = model(goal)
        repr = torch.cat((repr, picker_state), dim=1)
        goal_repr = torch.cat((goal_repr, goal_picker_state), dim=1)
        reprs.append(repr)
        actions.append(action)
        goals.append(goal_repr)
    reprs = torch.cat(reprs, dim=0)
    actions = torch.cat(actions, dim=0)
    goals = torch.cat(goals, dim=0)
    assert reprs.shape[0] == actions.shape[0] == goals.shape[0]
    print('[INFO] Number of samples: {}'.format(reprs.shape[0]))
    print('[INFO] Representation shape: {}'.format(reprs.shape))
    print('[INFO] Action shape: {}'.format(actions.shape))
    print('[INFO] Goal shape: {}'.format(goals.shape))
    torch.save(reprs, './model/reprs.pt')
    torch.save(actions, './model/actions.pt')
    torch.save(goals, './model/goals.pt')