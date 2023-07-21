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

def load_model(path='/home/hnguyen/cloth_smoothing/equiRL/learner.pth'):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.fc = nn.Identity()
    model.eval()
    return model

# Define a custom dataset to load the NPY files
class NPYDataset(Dataset):
    def __init__(self, npy_files, columns=['NPY_Path']):
        self.data = pd.read_csv(npy_files, usecols=columns)['NPY_Path'].values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        npy_file = self.data[idx]
        frame = np.load(npy_file, allow_pickle=True)
        return frame

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        obs, action, next_obs = self.data_list[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), torch.tensor(next_obs, dtype=torch.float32)

def random_transform_image_rgbd(image, transform_custom_rgb):
    rgb = image[:, :3, :, :]
    depth = image[:, 3, :, :]
    rgb = torch.stack([transform_custom_rgb(img) for img in rgb])
    image = torch.cat((rgb, depth.unsqueeze(1)), dim=1)
    return image

if __name__ == '__main__':
    # Load the dataset
    dataloaders = []
    batch_size = 512
    transform_custom_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor()
])
    dataset_play = NPYDataset('/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/play.csv')
    dataset_demo = NPYDataset('/home/hnguyen/cloth_smoothing/equiRL/data/equi/video/demo.csv')
    for dataset in [dataset_play, dataset_demo]:
        for frame in dataset:
            for i in range(len(frame)):
                obs = frame[i][0]
                next_obs = frame[i][3]
                # add to dataloader
                dataloaders.append([obs, next_obs])

    my_dataloader = MyDataset(dataloaders)
    dataloader = DataLoader(my_dataloader, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet18(pretrained=False)
    resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    learner = BYOL(
        resnet,
        image_size=128,
        hidden_layer=-2,
    )
    learner = learner.to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    losses = []
    for _ in range (100):
        l = 0
        for idx, data in tqdm(enumerate(dataloader)):
            obs = data[0].view(-1, 4, 128, 128)
            obs /= 255.0
            obs = obs.to(device)
            next_obs = data[1].view(-1, 4, 128, 128)
            next_obs /= 255.0
            next_obs = next_obs.to(device)
            loss = learner(obs, next_obs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            l += loss.item()
        losses.append(l)
        print('[INFO] Epoch: {}, Loss: {}'.format(_, l))

    # Save the model
    torch.save(learner.state_dict(), 'learner.pth')

    # Plot the losses
    plt.plot(losses)
    plt.savefig('losses.png')


