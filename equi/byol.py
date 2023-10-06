import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms as T

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        # DEFAULT_AUG = torch.nn.Sequential(
        #     RandomApply(
        #         T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        #         p = 0.3
        #     ),
        #     T.RandomGrayscale(p=0.2),
        #     T.RandomHorizontalFlip(),
        #     RandomApply(
        #         T.GaussianBlur((3, 3), (1.0, 2.0)),
        #         p = 0.2
        #     ),
        #     T.RandomResizedCrop((image_size, image_size)),
        #     T.Normalize(
        #         mean=torch.tensor([0.485, 0.456, 0.406]),
        #         std=torch.tensor([0.229, 0.224, 0.225])),
        # )

        # self.augment1 = default(augment_fn, DEFAULT_AUG)
        # self.augment2 = default(augment_fn2, self.augment1)
        self.augment1 = T.Compose([
            RandomChangeBackgroundRGBD(p=0.5),
            RandomColorJitterRGBD(p=0.5),
            RandomGrayScaleRGBD(p=0.3),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        ])
        self.augment2 = T.Compose([
            RandomChangeBackgroundRGBD(p=0.5),
            RandomColorJitterRGBD(p=0.5),
            RandomGrayScaleRGBD(p=0.3),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        ])
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randint(0, 255, (2, 4, image_size, image_size), device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        image_one, image_two = self.augment1(x), self.augment2(x)
        image_one = image_one.to(torch.float32) / 127.5 - 1
        image_two = image_two.to(torch.float32) / 127.5 - 1

        online_proj_one, representation_one = self.online_encoder(image_one)
        online_proj_two, representation_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two
        final_loss = loss.mean()
        return final_loss
    
# class Augmentation RGBD images
class RandomGrayScaleRGBD(object):
    def __init__(self, p=0.5):
        self.p = p
        self.aug = T.Grayscale(num_output_channels=3)

    def __call__(self, img):
        if random.random() < self.p:
            img_rgb = img[:, :3, :, :]
            img_depth = img[:, 3:, :, :]
            img_gray = self.aug(img_rgb)
            img = torch.cat([img_gray, img_depth], dim=1)
        return img

class RandomColorJitterRGBD(object):
    def __init__(self, p=0.5):
        self.p = p
        self.aug = T.ColorJitter(0.5, 0.5, 0.5, 0.2)

    def __call__(self, img):
        if random.random() < self.p:
            img_rgb = img[:, :3, :, :]
            img_depth = img[:, 3:, :, :]
            img_rgb = self.aug(img_rgb)
            img = torch.cat([img_rgb, img_depth], dim=1)
        return img

class RandomChangeBackgroundRGBD(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            img_rgb = img[:, :3, :, :]
            img_depth = img[:, 3:, :, :]
            mask = (img_depth >= 253).squeeze(1)
            img_rgb_masked = img_rgb.clone()
            random_color = torch.rand(3)
            for i in range(3):
                img_rgb_masked[:, i, :, :][mask] = int(random_color[i] * 255)
            img = torch.cat([img_rgb_masked, img_depth], dim=1)
        return img
    
class TCN(object):
    "Time-constrastive network"
    def __init__(self):
        self.reg_lambda = 0.002

    def npairs_loss(self, embeddings_anchor, embeddings_positive, step):
        "Returns n-pairs loss for a single sequence"
        reg_anchor = torch.mean(torch.sum(torch.square(embeddings_anchor), dim=1))
        reg_positive = torch.mean(torch.sum(torch.square(embeddings_positive), dim=1))
        l2loss = 0.25 * self.reg_lambda * (reg_anchor + reg_positive)
        
        # normalize embeddings_anchor and embeddings_positive
        # embeddings_anchor = embeddings_anchor / torch.norm(embeddings_anchor, dim=1, keepdim=True)
        # embeddings_positive = embeddings_positive / torch.norm(embeddings_positive, dim=1, keepdim=True)
        # get per pair         
        similarity_matrix = torch.matmul(embeddings_anchor, embeddings_positive.transpose(0, 1))

        step = Variable(step, requires_grad=False)
        step = step.float()
        weight = torch.exp(-torch.abs(step.view(-1, 1) - step))
        weight_sum_row = torch.sum(weight, dim=1)
        weight /= weight_sum_row.view(-1, 1)
        weight = weight.detach()

        ms = torch.sum(torch.exp(similarity_matrix), dim=1)
        xent_loss = -torch.log(torch.exp(similarity_matrix) / ms.view(-1, 1)) * weight
        xent_loss = torch.mean(torch.sum(xent_loss, dim=1))
        return xent_loss + l2loss
        
    def compute_loss(self, embeddings_anchor, embeddings_positive, steps):
        "Returns n-pairs loss for a batch of sequences"
        batch_loss = []
        for i in range(embeddings_anchor.shape[0]):
            batch_loss.append(self.npairs_loss(embeddings_anchor[i], embeddings_positive[i], steps[i]))
        return torch.mean(torch.stack(batch_loss, dim=0))
