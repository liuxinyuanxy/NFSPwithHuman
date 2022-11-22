from ai.nfsp.average_policy import *
from ai.nfsp.best_response_policy import *

import os
import enum
import json
from collections import namedtuple
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

MODE = enum.Enum("mode", "best_response average_policy")


class NFSP(object):
    def __init__(self,
                 config_dir,
                 name,
                 state_representation_size,
                 num_actions,
                 is_evaluation,
                 sl_loss_backward = None,
                 device = None
                ):

        config_path = os.path.join(config_dir, name)+".json"
        with open(config_path, 'r') as config_file:
            self._config = json.load(config_file)

        self._state_representation_size = state_representation_size
        self._num_actions = num_actions

        self._anticipatory_param = self._config["anticipatory_param"]
        self._learn_every = self._config["learn_every"]

        self._is_evaluation = is_evaluation

        if device is None:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

        sl_kwargs = {
            "device" : self._device,
            "state_representation_size" : self._state_representation_size,
            "num_actions" : self._num_actions
        }
        if not sl_loss_backward is None:
            sl_kwargs.update({'loss_backward':sl_loss_backward})
        rl_kwargs = {
            "device" : self._device,
            "state_representation_size" : self._state_representation_size,
            "num_actions" : self._num_actions
        }

        sl_kwargs.update(self._config['sl_kwargs'])
        rl_kwargs.update(self._config['rl_kwargs'])

        self._checkpoint_path = os.path.join(self._config['checkpoint_dir'], self._config['name'])+".pth"


        # Step counter to keep track of learning.
        self._step_counter = 0

        # Keep track of the last training loss achieved in an update step.
        self._last_rl_loss_value = None
        self._last_sl_loss_value = None

        self._best_response_policy = BestResponsePolicy(**rl_kwargs)
        self._avg_policy = AveragePolicy(**sl_kwargs)

        if self._is_evaluation:
            self.load()
        else:
            self._best_response_policy.initial_parameter()
            self._avg_policy.initial_parameter()

        self._sample_episode_policy()
        self._prev_info_state = None
        self._prev_action = None


    def _sample_episode_policy(self):
        if np.random.rand() < self._anticipatory_param:
            self._mode = MODE.best_response
        else:
            self._mode = MODE.average_policy    

    def step(self, info_state, legal_actions, reward, done, is_evaluation = None):
        """Returns the action to be taken and updates the Q-networks if needed.
        Args:
            obs: a 0-1 tensor translated from state.
            is_evaluation: bool, whether this is a training or evaluation call.
        Returns:
            A `rl_agent.StepOutput` containing the action probs and chosen action.
        """

        if is_evaluation == None:
            is_evaluation = self._is_evaluation

        if done == 1:
            action = self._num_actions-1 #call
        else:
            if self._mode == MODE.best_response: #and not is_evaluation:
                probs = self._best_response_policy.predict(info_state, legal_actions)
                action = np.random.choice(self._num_actions, p = probs)
                self._avg_policy.add_behavior_tuple(info_state, action)

            else:
                probs = self._avg_policy.act(info_state, legal_actions)
                action = np.random.choice(self._num_actions, p = probs)

        if done == 1:
            self._sample_episode_policy()

        if not is_evaluation:
            if not self._prev_info_state is None:
                self._best_response_policy.add_transition(self._prev_info_state, self._prev_action, reward, info_state, legal_actions, done)
        
            self._step_counter += 1
            if self._step_counter % self._learn_every == 0:
                self._last_sl_loss_value = self._avg_policy.train()
                self._last_rl_loss_value = self._best_response_policy.train()

            # Prepare for the next episode.
            if done == 1:
                self._prev_info_state = None
                self._prev_action = None
            else:
                self._prev_info_state = info_state
                self._prev_action = action

        return action

        


    def save(self, checkpoint_path = None):
        """Saves the average policy network and the inner RL agent's q-network.
        Note that this does not save the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.
        Args:
        checkpoint_dir: directory where checkpoints will be saved.
        """
        save_path = checkpoint_path if checkpoint_path else self._checkpoint_path
        checkpoint = {}
        checkpoint.update(self._best_response_policy.get_checkpoint())
        checkpoint.update(self._avg_policy.get_checkpoint())
        torch.save(checkpoint, save_path)

    def load(self, checkpoint_path = None):
        """Restores the average policy network and the inner RL agent's q-network.
        Note that this does not restore the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.
        Args:
            checkpoint_dir: directory from which checkpoints will be restored.
        """
        load_path = checkpoint_path if checkpoint_path else self._checkpoint_path
        checkpoint = torch.load(load_path, map_location = self._device)
        self._best_response_policy.load(checkpoint)
        self._avg_policy.load(checkpoint)