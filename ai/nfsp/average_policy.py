import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
import numpy as np

BehaviorTuple = namedtuple(
    "BehaviorTuple", "info_state action")
class AveragePolicy:
    def __init__(self,
                 device,
                 state_representation_size,
                 num_actions,
                 hidden_layers_sizes,
                 optimizer_str,
                 batch_size,
                 min_buffer_size_to_learn,
                 learning_rate,
                 reservoir_buffer_capacity,
                 loss_backward = None
                 ):

        self._state_representation_size = state_representation_size
        self._layer_sizes = hidden_layers_sizes
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._min_buffer_size_to_learn = min_buffer_size_to_learn
        self._device = device
        self._sl_memory = ReservoirBuffer(reservoir_buffer_capacity)

        self._module = SLModule(state_representation_size, self._layer_sizes, num_actions).to(self._device)
        self._module.eval()

        if optimizer_str == "adam":
            self._optimizer = torch.optim.Adam(
                self._module.parameters(), lr=learning_rate)
        elif optimizer_str == "sgd":
            self._optimizer = torch.optim.SGD(
                self._module.parameters(), lr=learning_rate)

        
        self._loss_backward = self._default_loss_backward if loss_backward is None else loss_backward

    def _cross_entropy_loss(self, info_states, actions):
        return F.cross_entropy(self._module(info_states),
                            actions)

    def initial_parameter(self):
        for p in self._module.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

    def act(self, info_state, legal_actions):
        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(device = self._device)

        with torch.no_grad():
            rough_module_output = self._module(info_state)
            rough_module_output[0][~legal_actions] = -np.inf
            action_probs = F.softmax(rough_module_output, dim = 1).cpu().numpy().flatten()

        return action_probs

    
    def _default_loss_backward(info_states, output, actions):
        loss = F.cross_entropy(output, actions)
        loss.backward()
        return loss

    def train(self):
        if (len(self._sl_memory) < self._batch_size or 
            len(self._sl_memory) < self._min_buffer_size_to_learn):
            return None


        behavior_tuples = self._sl_memory.sample(self._batch_size)
        info_states = torch.Tensor([t.info_state for t in behavior_tuples]).float().to(device = self._device)
        actions = torch.Tensor([t.action for t in behavior_tuples]).long().to(device = self._device)

        self._optimizer.zero_grad()
        self._module.train()
        #loss = F.cross_entropy(self._module(info_states), actions)
        outputs = self._module(info_states)
        loss = self._loss_backward(info_states, outputs, actions)
        self._optimizer.step()
        self._module.eval()

        return loss.detach()

    def add_behavior_tuple(self, info_state, action):
        """Adds the new transition using `time_step` to the reservoir buffer.
        Transitions are in the form (time_step, agent_output.probs, legal_mask).
        Args:
            time_step: an instance of rl_environment.TimeStep.
            agent_output: an instance of rl_agent.StepOutput.
        """
        behavior_tuple = BehaviorTuple(
            info_state=info_state,
            action=action)
        self._sl_memory.add(behavior_tuple)

    def get_checkpoint(self):
        return {
            'sl_network' : self._module.state_dict()
        }

    def load(self, checkpoint):
        self._module.load_state_dict(checkpoint['sl_network'])

class SLModule(nn.Module):
    def __init__(self, state_representation_size, hidden_layer_sizes, action_size):
        ''' Initialize the policy network.  It's just a bunch of ReLU

        Args:

        '''
        super(SLModule, self).__init__()

        # set up mlp w/ relu activations
        layer_dims = [state_representation_size] +hidden_layer_sizes+ [action_size]
        mlp = [nn.Flatten()]
        # mlp.append(nn.BatchNorm1d(layer_dims[0]))
        # for i in range(len(layer_dims)-1):
        #     mlp.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        #     if i != len(layer_dims) - 2: # all but final have relu
        #         mlp.append(nn.ReLU())
        # self.mlp = nn.Sequential(*mlp)

        for i in range(len(layer_dims)-1):
            mlp.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims) - 2:
                mlp.append(nn.BatchNorm1d(layer_dims[i+1]))
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, s):
        return self.mlp(s)


class ReservoirBuffer(object):
    ''' Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    '''

    def __init__(self, reservoir_buffer_capacity):
        ''' Initialize the buffer.
        '''
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        ''' Potentially adds `element` to the reservoir buffer.

        Args:
            element (object): data to be added to the reservoir buffer.
        '''
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        ''' Returns `num_samples` uniformly sampled from the buffer.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            An iterable over `num_samples` random elements of the buffer.

        Raises:
            ValueError: If there are less than `num_samples` elements in the buffer
        '''
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                    num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        ''' Clear the buffer
        '''
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)