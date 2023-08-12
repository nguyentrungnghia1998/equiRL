import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from byol import BYOL
from tqdm import tqdm

class MLP_BC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, last_activation='tanh'):
        super(MLP_BC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        if last_activation == 'tanh':
            self.last = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.last(x)
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
            rnn_input_dim,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type="LSTM", # [LSTM, GRU]
            rnn_is_bidirectional=False
            ):
        """
        rnn_input_dim (int): RNN input dimension
        rnn_hidden_dim (int): RNN hidden dimension
        rnn_num_layers (int): number of RNN layers
        rnn_type (str): [LSTM, GRU]
        rnn_is_bidirectional: whether using bidirectional
        """
        # super(RNN_MIMO_MLP).__init__()
        super(RNN_MIMO_MLP, self).__init__()
        self.rnn_is_bidirectional = rnn_is_bidirectional
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_directions = 2 if rnn_is_bidirectional else 1
        self.rnn_num_layers = rnn_num_layers
        rnn_output_dim = self.num_directions * rnn_hidden_dim

        self.mlp = MLP_BC(
            input_dim=rnn_output_dim,
            hidden_dim=256,
            output_dim=8
        )

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=rnn_is_bidirectional
        )

    def get_rnn_init_state(self, batch_size, device):
        """
        Get initial state for RNN
        """
        h0 = torch.zeros(self.rnn_num_layers * self.num_directions , batch_size, self.rnn_hidden_dim).to(device)
        if self.rnn_type == "LSTM":
            c0 = torch.zeros(self.rnn_num_layers * self.num_directions, batch_size, self.rnn_hidden_dim).to(device)
            return h0, c0
        else:
            return h0
        
    def forward(self, inputs, rnn_init_state=None, return_state=False):
        """
        inputs: [B, T, D]
        rrn_init_state: initialize to zero if set to None
        return_state: whether to return the hidden state

        Retuns:
            outputs: outputs of the per step net
            rnn_state: return rnn state if return_state is True
        """
        assert inputs.dim() == 3, "Input dimension should be 3"
        batch_size, seq_len, _ = inputs.size()
        if rnn_init_state is None:
            rnn_init_state = self.get_rnn_init_state(batch_size, inputs.device)
        outputs, rnn_state = self.rnn(inputs, rnn_init_state)
        outputs = self.mlp(outputs)
        if return_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(self, inputs, rnn_state):
        """
        inputs: [B, D]
        rnn_state: previous rnn state

        Retuns:
            outputs: outputs of the per step net
            rnn_state: return rnn state if return_state is True
        """
        assert inputs.dim() == 2, "Input dimension should be 2"
        outputs, rnn_state = self.forward(inputs.unsqueeze(1), rnn_state, True)
        return outputs[:, 0], rnn_state

def load_model(model, path='/home/hnguyen/cloth_smoothing/equiRL/learn_repr.pt', representation=True):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    if representation:
        model.fc = nn.Identity()
    model.eval()
    return model

# Define a custom dataset to load the NPY files
class NPYDataset(Dataset):
    def __init__(self, npy_files, columns=['NPY_Path'], play=True):
        self.data = pd.read_csv(npy_files, usecols=columns).values
        self.play = play
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.play:
            npy_file = self.data[idx][0]
            frame = np.load(npy_file, allow_pickle=True)
            return frame
        else:
            length, final_step_str, npy_file = self.data[idx]
            frame = np.load(npy_file, allow_pickle=True)
            final_step_str = final_step_str.replace('[', '').replace(']', '').split(',')
            final_step = [int(i) for i in final_step_str]
            return length, final_step, frame

class RNN_dataset(Dataset):
    def __init__(self, dataset, max_length=201):
        super(RNN_dataset, self).__init__()
        self.dataset = dataset

        self.trajs = []
        for length, final_step, frame in dataset:
            j = 0
            traj = []
            for k in range(1, len(frame)+1):
                obs = frame[k-1][0]
                picker_state = frame[k-1][5]
                next_picker_state = frame[k-1][6]
                goal_obs = frame[final_step[j]-1][3]
                goal_picker_state = frame[final_step[j]-1][6]
                step = final_step[j] - k + 1
                action = frame[k-1][1]

                obs = obs.to(dtype=torch.float32) / 255.0
                picker_state = torch.tensor(picker_state, dtype=torch.float32)
                action = torch.tensor(action, dtype=torch.float32)
                goal_obs = goal_obs.to(dtype=torch.float32) / 255.0
                goal_picker_state = torch.tensor(goal_picker_state, dtype=torch.float32)
                step = torch.tensor(step, dtype=torch.float32)
                
                if k == final_step[j]:
                    j += 1
                traj.append([obs, picker_state, action, goal_obs, goal_picker_state, step])
            self.trajs.append(traj)
    
    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

class Dataset_Repr(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        obs, next_obs = self.data_list[idx]
        obs = obs.to(dtype=torch.float32)
        next_obs = next_obs.to(dtype=torch.float32)
        return obs, next_obs

class Dataset_trajectories(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        traj = self.data_list[idx]
        return traj
    
def train_representation_learning(model, dataloader, device, img_size=224, num_epochs=100):
    losses = []
    learner = BYOL(
        model,
        image_size=img_size,
        hidden_layer=-2,
    )
    learner = learner.to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    for _ in range (num_epochs):
        l = 0
        for idx, data in tqdm(enumerate(dataloader)):
            obs = data[0].view(-1, 4, img_size, img_size)
            obs /= 255.0
            obs = obs.to(device)
            next_obs = data[1].view(-1, 4, img_size, img_size)
            next_obs /= 255.0
            next_obs = next_obs.to(device)
            loss = learner(obs, next_obs)
            l += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        losses.append(l)
        if (_ + 1) % 10 == 0:   
            print('[INFO] Epoch: {}, Loss: {}'.format(_+1, l))

    # Save the model
    torch.save(model.state_dict(), 'learned_repr.pt')

    # Plot the losses
    plt.plot(losses)
    plt.title('Representation Learning Losses (BYOL)')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('representation_losses.png')
    plt.close()

def train_BC(model, bc_model, expert_dataloader, device, img_size=224, num_epochs=100, k=32):
    model.eval()
    bc_model.train()
    bc_opt = torch.optim.Adam(bc_model.parameters(), lr=3e-4)
    bc_losses = []
    best_loss = np.inf
    for _ in range(num_epochs):
        l = 0
        for traj in expert_dataloader:
            for i in range(0, len(traj), k):
                obs = torch.stack([t[0] for t in traj[i:i+k]], dim=0).view(-1, 4, img_size, img_size).to(device)
                picker_state = torch.stack([t[1] for t in traj[i:i+k]], dim=0).view(-1, 2).to(device)
                action = torch.stack([t[2] for t in traj[i:i+k]], dim=0).view(-1, 8).to(device)
                with torch.no_grad():
                    repr = model(obs)
                # Concatenate the picker state
                repr = torch.cat((repr, picker_state), dim=1)
                pred = bc_model(repr)
                loss = nn.MSELoss()(pred, action)
                bc_opt.zero_grad()
                loss.backward()
                bc_opt.step()
                l += loss.item()

        bc_losses.append(l)
        if (_ + 1) % 10 == 0:
            print('[INFO] Epoch: {}, Loss: {}'.format(_+1, l))
        if l < best_loss:
            best_loss = l
            torch.save(bc_model.state_dict(), 'bc_model.pt')

    # Plot the losses
    plt.figure()
    plt.plot(bc_losses)
    plt.title('BC Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('bc_losses.png')
    plt.close()

def train_BC_RNN(model, bc_rnn_model, expert_dataloader, device, img_size=224, num_epochs=100, k=32):
    # model.eval()
    model.train()
    bc_rnn_opt = torch.optim.Adam(bc_rnn_model.parameters(), lr=3e-4)
    bc_rnn_losses = []
    best_loss = np.inf
    for _ in range(num_epochs):
        l = 0
        for traj in expert_dataloader:
            repr_traj = []
            action_traj = []
            for i in range(0, len(traj), k):
                obs = torch.stack([t[0] for t in traj[i:i+k]], dim=0).view(-1, 4, img_size, img_size).to(device)
                picker_state = torch.stack([t[1] for t in traj[i:i+k]], dim=0).view(-1, 2).to(device)
                action = torch.stack([t[2] for t in traj[i:i+k]], dim=0).view(-1, 8).to(device)
                # with torch.no_grad():
                repr = model(obs)
                # Concatenate the picker state
                repr = torch.cat((repr, picker_state), dim=1)
                repr_traj.append(repr)
                action_traj.append(action)
            repr_traj = torch.cat(repr_traj, dim=0).unsqueeze(0)
            action_traj = torch.cat(action_traj, dim=0)

            # Train the RNN
            action_pred = bc_rnn_model(repr_traj).squeeze(0)
            loss = nn.MSELoss()(action_pred, action_traj)
            bc_rnn_opt.zero_grad()
            loss.backward()
            bc_rnn_opt.step()
            l += loss.item()

        bc_rnn_losses.append(l)
        if (_ + 1) % 10 == 0:
            print('[INFO] Epoch: {}, Loss: {}'.format(_+1, l))
        if l < best_loss:
            best_loss = l
            torch.save(bc_rnn_model.state_dict(), 'bc_rnn_model_0_repr.pt')
            torch.save(model.state_dict(), 'learned_0_repr.pt')

    # Plot the losses
    plt.figure()
    plt.plot(bc_rnn_losses)
    plt.title('BC RNN Losses 0 repr')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.savefig('bc_rnn_losses.png')
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
    # Load the dataset
    dataloaders = []
    batch_size = 300
    dataset_play = NPYDataset(npy_files='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/play.csv', columns=['NPY_Path'], play=True)
    for frame in dataset_play:
        for i in range(len(frame)):
            obs = frame[i][0]
            next_obs = frame[i][3]
            # add to dataloader
            dataloaders.append([obs, next_obs])
    dataset_demo = NPYDataset(npy_files='/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv', columns=['Length', 'Final_step', 'NPY_Path'], play=False)
    for length, final_step, frame in dataset_demo:
        for i in range(len(frame)):
            obs = frame[i][0]
            next_obs = frame[i][3]
            # add to dataloader
            dataloaders.append([obs, next_obs])
    my_dataloader = Dataset_Repr(dataloaders)
    dataloader = DataLoader(my_dataloader, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the model
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity()
    model.to(device)
    # train_representation_learning(model, dataloader, device, img_size=168, num_epochs=100)
    # Load the dataset and build dataloader for BC

    ep_dataset = RNN_dataset(dataset_demo)
    expert_dataloader  = DataLoader(ep_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load the Representation model
    # model = load_model(model, path='/home/hnguyen/cloth_smoothing/equiRL/learned_repr.pt', representation=True).to(device)
    # model.eval()

    # Define the BC model
    # bc_model = MLP_BC(input_dim=514, output_dim=8, hidden_dim=256).to(device)
    # train_BC(model, bc_model, expert_dataloader, device, img_size=168, num_epochs=100)
    # bc_model = load_model(bc_model, path='/home/hnguyen/cloth_smoothing/equiRL/bc_model.pt', representation=False)
    # Define BC RNN model
    bc_rnn_model = RNN_MIMO_MLP(rnn_input_dim=514, rnn_hidden_dim=1024, rnn_num_layers=2, rnn_type="LSTM", rnn_is_bidirectional=False).to(device)
    train_BC_RNN(model, bc_rnn_model, expert_dataloader, device, img_size=168, num_epochs=100)
    bc_rnn_model = load_model(bc_rnn_model, path='/home/hnguyen/cloth_smoothing/equiRL/bc_rnn_model_0_repr.pt', representation=False)
    exit()
    
    # Define the dynamical model
    # dynamical_model = Dynamical_Model(input_dim=514, output_dim=1, hidden_dim=256).to(device)
    # train_dynamical_step(model, dynamical_model, expert_dataloader, device, img_size=168, num_epochs=100)
    # dynamical_model = load_model(dynamical_model, path='/home/hnguyen/cloth_smoothing/equiRL/dynamical_model.pt', representation=False)


    # Save the representation and action
    reprs = []
    actions = []
    goals = []
    for traj in expert_dataloader:
        for i in range(0, len(traj), 100):
            obs = torch.stack([t[0] for t in traj[i:i+100]], dim=0).view(-1, 4, 168, 168).to(device)
            picker_state = torch.stack([t[1] for t in traj[i:i+100]], dim=0).view(-1, 2).to(device)
            action = torch.stack([t[2] for t in traj[i:i+100]], dim=0).view(-1, 8).to(device)
            goal_obs = torch.stack([t[3] for t in traj[i:i+100]], dim=0).view(-1, 4, 168, 168).to(device)
            goal_picker_state = torch.stack([t[4] for t in traj[i:i+100]], dim=0).view(-1, 2).to(device)
            with torch.no_grad():
                repr = model(obs)
                goal_repr = model(goal_obs)
            # Concatenate the picker state
            repr = torch.cat((repr, picker_state), dim=1)
            reprs.append(repr)
            actions.append(action)
            goal_repr = torch.cat((goal_repr, goal_picker_state), dim=1)
            goals.append(goal_repr)

    reprs = torch.cat(reprs, dim=0)
    actions = torch.cat(actions, dim=0)
    goals = torch.cat(goals, dim=0)
    assert reprs.shape[0] == actions.shape[0] == goals.shape[0]
    print('[INFO] Number of samples: {}'.format(reprs.shape[0]))
    print('[INFO] Representation shape: {}'.format(reprs.shape))
    print('[INFO] Action shape: {}'.format(actions.shape))
    print('[INFO] Goal shape: {}'.format(goals.shape))
    torch.save(reprs, 'reprs.pt')
    torch.save(actions, 'actions.pt')
    torch.save(goals, 'goals.pt')