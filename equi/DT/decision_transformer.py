import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import transformers

# from model import TrajectoryModel
from DT.model import TrajectoryModel
from DT.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            load_obs_pretrained=False,
            finetune_obs_pretrained=False,
            path_pretrained=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.load_obs_pretrained = load_obs_pretrained
        self.finetune_obs_pretrained = finetune_obs_pretrained
        self.embed_obs = models.resnet18(pretrained=False)
        self.embed_obs.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.load_obs_pretrained:
            assert path_pretrained is not None, "Path to pretrained model is None"
            self.embed_obs.load_state_dict(torch.load(path_pretrained))
        self.embed_obs.fc = nn.Identity()
        if self.finetune_obs_pretrained is False:
            for param in self.embed_obs.parameters():
                param.requires_grad = False

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, obs, picker_state, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length, channel, height, width = obs.size()
        if self.finetune_obs_pretrained is False:
            with torch.no_grad():
                repr = self.embed_obs(obs.view(-1, channel, height, width))
        else:
            repr = self.embed_obs(obs.view(-1, channel, height, width))
        repr = repr.view(batch_size, seq_length, -1)
        states = torch.cat((repr, picker_state), dim=2)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(obs.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        # return state_preds, action_preds, return_preds
        return action_preds

class Factorize_DT_Transformer(TrajectoryModel):
    """
    Factorize inputs and outputs
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            load_obs_pretrained=False,
            finetune_obs_pretrained=False,
            path_pretrained=None,
            predict_next_state=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.load_obs_pretrained = load_obs_pretrained
        self.finetune_obs_pretrained = finetune_obs_pretrained
        self.predict_picker_state = predict_next_state
        self.embed_obs = models.resnet18(pretrained=False)
        self.embed_obs.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.load_obs_pretrained:
            assert path_pretrained is not None, "Path to pretrained model is None"
            self.embed_obs.load_state_dict(torch.load(path_pretrained))
        self.embed_obs.fc = nn.Identity()
        if self.finetune_obs_pretrained is False:
            for param in self.embed_obs.parameters():
                param.requires_grad = False
        
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_picker_state = nn.Embedding(4, hidden_size)
        # self.embed_picker_state = torch.nn.Linear(2, hidden_size)

        self.embed_action_pos = torch.nn.Linear(3, hidden_size)
        self.embed_action_pp = nn.Embedding(2, hidden_size)
        # self.embed_action_pp = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # predict actions, next_states
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_picker_state = torch.nn.Linear(hidden_size, 2)
        self.predict_pos = nn.Sequential(torch.nn.Linear(hidden_size, 3), nn.Tanh())
        self.predict_pp = torch.nn.Linear(hidden_size, 1)

    def forward(self, obs, picker_state, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length, channel, height, width = obs.size()
        if self.finetune_obs_pretrained is False:
            with torch.no_grad():
                repr = self.embed_obs(obs.view(-1, channel, height, width))
        else:
            repr = self.embed_obs(obs.view(-1, channel, height, width))
        repr = repr.view(batch_size, seq_length, -1)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(obs.device)
        # code picker state [0, 0] -> 0, [0, 1] -> 1, [1, 0] -> 2, [1, 1] -> 3
        picker_state_code = torch.zeros((batch_size, seq_length)).to(obs.device)
        picker_state_code[(picker_state[..., 0] == 0) & (picker_state[..., 1] == 1)] = 1
        picker_state_code[(picker_state[..., 0] == 1) & (picker_state[..., 1] == 0)] = 2
        picker_state_code[(picker_state[..., 0] == 1) & (picker_state[..., 1] == 1)] = 3
        picker_state_code = picker_state_code.long()
        # split actions
        actions_pos_1, actions_pp_1 = actions[..., :3], actions[..., 3:4]
        actions_pos_2, actions_pp_2 = actions[..., 4:7], actions[..., 7:8]
        actions_pp_1 = (actions_pp_1 + 1) / 2
        actions_pp_2 = (actions_pp_2 + 1) / 2
        actions_pp_1 = actions_pp_1.squeeze(-1).long()
        actions_pp_2 = actions_pp_2.squeeze(-1).long()
        # embed each modality with a different head
        state_embeddings = self.embed_state(repr)
        picker_state_embeddings = self.embed_picker_state(picker_state_code)
        action_pos_embeddings_1 = self.embed_action_pos(actions_pos_1)
        action_pp_embeddings_1 = self.embed_action_pp(actions_pp_1)
        action_pos_embeddings_2 = self.embed_action_pos(actions_pos_2)
        action_pp_embeddings_2 = self.embed_action_pp(actions_pp_2)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        state_embeddings = state_embeddings + time_embeddings
        picker_state_embeddings = picker_state_embeddings + time_embeddings
        action_pos_embeddings_1 = action_pos_embeddings_1 + time_embeddings
        action_pp_embeddings_1 = action_pp_embeddings_1 + time_embeddings
        action_pos_embeddings_2 = action_pos_embeddings_2 + time_embeddings
        action_pp_embeddings_2 = action_pp_embeddings_2 + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, picker_state_embeddings, action_pos_embeddings_1, action_pp_embeddings_1, action_pos_embeddings_2, action_pp_embeddings_2), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 7*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 7*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), picker_state (2), action_pos_1 (3), action_pp_1 (4), action_pos_2 (5), action_pp_2 (6)
        x = x.reshape(batch_size, seq_length, 7, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        action_preds_pos_1 = self.predict_pos(x[:,2])
        action_preds_pp_1 = self.predict_pp(x[:,3])
        action_preds_pos_2 = self.predict_pos(x[:,4])
        action_preds_pp_2 = self.predict_pp(x[:,5])
        if self.predict_picker_state:
            next_state_preds = self.predict_state(x[:,6])
            return action_preds_pos_1, action_preds_pp_1, action_preds_pos_2, action_preds_pp_2, next_state_preds
        else:
            return action_preds_pos_1, action_preds_pp_1, action_preds_pos_2, action_preds_pp_2