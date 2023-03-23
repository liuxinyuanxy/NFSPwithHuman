from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition', "info_state action reward next_info_state done legal_action")

class BestResponsePolicy:
    def __init__(self,
                 device,
                 state_representation_size,
                 num_actions,
                 hidden_layers_sizes,
                 optimizer_str,
                 batch_size,
                 min_buffer_size_to_learn,
                 learning_rate,
                 replay_buffer_capacity,
                 update_target_network_every,
                 discount_factor,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps
                 ):

        self._state_representation_size = state_representation_size
        self._layer_sizes = hidden_layers_sizes
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._min_buffer_size_to_learn = min_buffer_size_to_learn
        self._device = device
        self._rl_memory = ReplayBuffer(replay_buffer_capacity)
        self._update_target_network_every = update_target_network_every
        self._discount_factor = discount_factor
        self._epsilons = np.linspace(epsilon_start, epsilon_end, int(epsilon_decay_steps))


        self._total_t = 0
        # Total training step
        self._train_t = 0

        # The epsilon decay scheduler

        self._q_network = RLModule(self._state_representation_size, self._layer_sizes,self._num_actions).to(self._device)
        self._target_q_network = deepcopy(self._q_network)
        self._q_network.eval()


        if optimizer_str == "adam":
            self._optimizer = torch.optim.Adam(self._q_network.parameters(), lr=learning_rate, weight_decay=0.001)
        elif optimizer_str == "sgd":
            self._optimizer = torch.optim.SGD(self._q_network.parameters(), lr=learning_rate, weight_decay=0.001)
        else:
            raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    def initial_parameter(self):
        for p in self._q_network.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self._target_q_network = deepcopy(self._q_network)


    def predict(self, info_state, legal_actions):
        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(self._device)

        with torch.no_grad():
            rough_qs = self._q_network(info_state).cpu().numpy().flatten()
        
        legal_qs = np.ones(self._num_actions, dtype = float)*(-float('inf'))
        legal_qs[legal_actions] = rough_qs[legal_actions]

        epsilon = self._epsilons[min(self._total_t, self._epsilons.size-1)]
        action_probs = np.zeros(self._num_actions, dtype=float)
        best_action = np.argmax(legal_qs)
        action_probs[legal_actions] = epsilon/sum(legal_actions)
        action_probs[best_action] += (1.0 - epsilon)

        return action_probs

    def add_transition(self,
                       info_state,
                       action,
                       reward,
                       next_info_state, 
                       legal_action, 
                       done):
        transition = Transition(info_state = info_state,
                                action = action, 
                                reward = reward, 
                                next_info_state = next_info_state, 
                                legal_action = legal_action,
                                done = done)
        self._rl_memory.add(transition)

    def train(self):
        if (len(self._rl_memory) < self._batch_size or
            len(self._rl_memory) < self._min_buffer_size_to_learn):
            return None

        self._q_network.train()

        transitions = self._rl_memory.sample(self._batch_size)
        info_states = torch.Tensor([t.info_state for t in transitions]).float().to(device = self._device)
        actions = torch.Tensor([t.action for t in transitions]).long().to(device = self._device)
        actions = actions.unsqueeze(dim = 1)
        rewards = torch.Tensor([t.reward for t in transitions]).float().to(device = self._device)
        next_info_states = torch.Tensor([t.next_info_state for t in transitions]).float().to(device = self._device)
        are_final_steps = torch.Tensor([t.done for t in transitions]).float().to(device = self._device)
        legal_actions = torch.Tensor([t.legal_action for t in transitions]).bool().to(device = self._device)

        q_values = self._q_network(info_states)

        target_q_values = torch.ones([self._batch_size, self._num_actions]).float().to(device = self._device) * (-float('inf'))
        target_q_values[legal_actions] = self._target_q_network(next_info_states).detach()[legal_actions]
        max_next_q = torch.max(target_q_values, dim = 1)[0]
        target = rewards + (1-are_final_steps) * self._discount_factor*max_next_q

        predictions = torch.gather(input = q_values, dim = 1, index = actions)

        loss = F.mse_loss(predictions.squeeze(1), target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._q_network.eval()

        if self._train_t % self._update_target_network_every == 0:
            self._target_q_network = deepcopy(self._q_network)
            print("\nINFO - Copied model parameters to target network.")

        self._train_t += 1

        return loss.detach()

    def get_checkpoint(self):
        return {
            'rl_network' : self._q_network.state_dict(),
            'rl_target_network' : self._target_q_network.state_dict()
        }

    def load(self, checkpoint):
        self._q_network.load_state_dict(checkpoint['rl_network'])
        self._target_q_network.load_state_dict(checkpoint['rl_target_network'])


class RLModule(torch.nn.Module):
    def __init__(self, state_representation_size, hidden_layer_sizes, action_size):
        super(RLModule, self).__init__()
        
        layer_dims = [state_representation_size] + hidden_layer_sizes + [action_size]

        mlp = [nn.Flatten()]
        # mlp.append(nn.BatchNorm1d(layer_dims[0]))

        # for i in range(len(layer_dims)-1):
        #     mlp.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias = True))
        #     if i != len(layer_dims) - 2: # all but final have tanh
        #         mlp.append(nn.Tanh())
        # self.mlp = nn.Sequential(*mlp)

        for i in range(len(layer_dims)-1):
            mlp.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias = True))
            if i != len(layer_dims) - 2:
                mlp.append(nn.BatchNorm1d(layer_dims[i+1]))
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, s):
        return self.mlp(s)

class ReplayBuffer(object):
    """ReplayBuffer of fixed size with a FIFO replacement policy.
    Stored transitions can be sampled uniformly.
    The underlying datastructure is a ring buffer, allowing 0(1) adding and
    sampling.
    """

    def __init__(self, replay_buffer_capacity):
        self._replay_buffer_capacity = replay_buffer_capacity
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        """Adds `element` to the buffer.
        If the buffer is full, the oldest element will be replaced.
        Args:
          element: data to be added to the buffer.
        """
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            self._data[self._next_entry_index] = element
            self._next_entry_index += 1
            self._next_entry_index %= self._replay_buffer_capacity

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.
        Args:
          num_samples: `int`, number of samples to draw.
        Returns:
          An iterable over `num_samples` random elements of the buffer.
        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)