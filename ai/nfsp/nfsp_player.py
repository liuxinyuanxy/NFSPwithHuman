
# Copyright 2021 Crise. All rights reserved.
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pypokerengine.players import BasePokerPlayer
from ai.nfsp.nfsp import NFSP
from ai.nfsp.nfsp_nolimit_holdem import *
import numpy as np 

class PyPokerEngineStateExtractor(StateExtractor):

    def __init__(self, owner):
        self._owner = owner

    @classmethod
    def _card2index(cls, card):
        return cls.CARD_TO_INDEX[card[::-1].capitalize()]

    @classmethod
    def _get_mychip_pot(cls, **kwargs):
        agent = kwargs['agent']
        round_state = kwargs['round_state']
        mychip = agent._round_initial_stack-[player['stack'] for player in round_state['seats'] if player['uuid'] == agent.uuid][0]
        pot = sum([pot['amount'] for pot in round_state['pot']['side']], round_state['pot']['main']['amount'])
        return mychip, pot
        
    @classmethod
    def _get_street(cls, **kwargs):
        round_state = kwargs['round_state']
        return round_state['street']
    
    @classmethod
    def _get_valid_action(cls, **kwargs):
        valid_actions = kwargs.get('valid_actions', None)
        if valid_actions == None:
            return None, 0
        round_state = kwargs['round_state']
        agent = kwargs['agent']
        stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == agent.uuid][0]
        paid_sum = max([0.0] + [action['amount'] for action in round_state['action_histories'][round_state['street']] if action['uuid'] == agent.uuid])

        result_valid_actions = {}
        for action in valid_actions:
            if action['action'] == 'fold':
                result_valid_actions['fold'] = True
            elif action['action'] == 'call':
                result_valid_actions['call'] = min(action['amount'] - paid_sum, stack)
            elif action['action'] == 'raise':
                result_valid_actions['raise'] = action['amount']['min']

        return result_valid_actions, stack


class NFSPPlayer(BasePokerPlayer):
    # class StateExtractor(object):
    #     '''
    #     提取状态信息
    #         info_state(float) info_cards 52位0-1手牌信息，
    #                           info_pot 2位下注信息(self, pot)， 
    #                           info_street 5位street信息，
    #                           info_actions num_action位合法动作信息
    #                                 fold
    #                                 call byamount
    #                                 can raise
    #                                 minraise by 
    #                                 all in by
    #         legal_action(bool) num_action位合法动作信息
    #         reward(float) 回报
    #         done(bool) 是否终局
    #     '''
    #     def __init__(self, owner):
    #         self._owner = owner

    #     def info_cards(self, cards):
    #         info_cards = np.zeros(52, dtype = float)
    #         idx = np.array([Card.from_str(card).to_id()-1 for card in cards])
    #         info_cards[idx] = 1.0
    #         return info_cards

    #     def info_pot(self, round_state):

    #         return np.array([
    #                             self._owner._round_initial_stack-[player['stack'] for player in round_state['seats'] if player['uuid'] == self._owner.uuid][0],
    #                             sum([pot['amount'] for pot in round_state['pot']['side']], round_state['pot']['main']['amount'])
    #                         ], 
    #                         dtype = float)

    #     def info_street(self, round_state):
    #         info_street = np.zeros(5, dtype = float)
    #         idx = STREET_PARSE.get(round_state['street'])
    #         if not idx is None:
    #             info_street[idx] = 1.0
    #         return info_street
        
    #     def info_actions_legal_actions(self, valid_actions = None, round_state = None):
    #         num_actions = self._owner._num_actions
    #         info_actions = np.zeros(5, dtype = float)
    #         legal_actions = np.zeros(num_actions, dtype = bool)

    #         if valid_actions:
    #             stack = [player['stack'] for player in round_state['seats'] if player['uuid'] == self._owner.uuid][0]
    #             paid_sum = max([0.0] + [action['amount'] for action in round_state['action_histories'][round_state['street']] if action['uuid'] == self._owner.uuid])
    #             for action in valid_actions:
    #                 if action['action'] == 'fold':
    #                     info_actions[0] = 1.0
    #                     legal_actions[num_actions - 2] = True
    #                 elif action['action'] == 'call':
    #                     info_actions[1] = min(action['amount'] - paid_sum, stack)
    #                     legal_actions[num_actions - 1] = True
    #                 elif action['action'] == 'raise':
    #                     info_actions[2] = 1.0
    #                     info_actions[3] = action['amount']['min'] - paid_sum
    #                     if action['amount']['min'] < action['amount']['max']:
    #                         legal_actions[0:num_actions-2] = True
    #                     else:
    #                         legal_actions[-3] = True
    #                 else:
    #                     raise "error action:" + action['action']
    #             info_actions[4] = stack

    #         return info_actions, legal_actions
            


    def __init__(self, config_dir, name, is_evaluation, device):

        self._is_evaluation = is_evaluation
        self._num_actions = PyPokerEngineStateExtractor.NUM_ACTIONS
        self._nfsp_policy = NFSP(config_dir,
                                 name,
                                 PyPokerEngineStateExtractor.STATE_REPRESENTATION_SIZE,
                                 self._num_actions,
                                 is_evaluation,
                                 sl_loss_backward,
                                 device
                                )

    def save(self):
        self._nfsp_policy.save()



    def declare_action(self, valid_actions, hole_card, round_state):
        info_cards = PyPokerEngineStateExtractor.info_cards(round_state['community_card'] + hole_card)
        info_pot = PyPokerEngineStateExtractor.info_pot(round_state = round_state, agent = self)
        info_street = PyPokerEngineStateExtractor.info_street(round_state = round_state)
        info_actions, legal_actions = PyPokerEngineStateExtractor.info_actions_legal_actions(valid_actions = valid_actions, round_state = round_state, agent = self)
        info_state = np.concatenate((info_cards, info_pot, info_street, info_actions))
        reward = 0.0
        done = 0.0
        action_code = self._nfsp_policy.step(info_state, legal_actions, reward, done, self._is_evaluation)
        if action_code == self._num_actions - 2:
            action = 'fold'
            amount = 0
        elif action_code == self._num_actions - 1:
            action = 'call'
            amount = [act['amount'] for act in valid_actions if act['action'] == action][0]
        elif action_code >=0 and action_code < self._num_actions - 2:
            action = 'raise'
            amount = PyPokerEngineStateExtractor.parse_raiseby(info_state, action_code)\
                + max([0.0] + [action['amount'] for action in round_state['action_histories'][round_state['street']] if action['uuid'] == self.uuid]) #paid_sum
        else:
            raise "error action"
        self._last_info_state = info_state
        return action, amount

    def receive_game_start_message(self, game_info):
        self.game_info = game_info

    def receive_round_start_message(self, round_count, hole_card, round_state):

        stack = sum([player['stack'] for player in round_state['seats'] if player['uuid']==self.uuid], 0.0) #使用sum防止本金不够不通知，这个list顶多一个元素
        ante_blind = sum([action['amount'] for action in round_state['action_histories'].get('preflop', []) if action['uuid'] == self.uuid], 0.0)
        self._round_initial_stack = stack + ante_blind
        info_cards = PyPokerEngineStateExtractor.info_cards(hole_card)
        info_pot = PyPokerEngineStateExtractor.info_pot(round_state = round_state, agent = self)
        info_street = PyPokerEngineStateExtractor.info_street(round_state = round_state)
        info_actions, _ = PyPokerEngineStateExtractor.info_actions_legal_actions()
        info_state = np.concatenate((info_cards, info_pot, info_street, info_actions))
        self._last_info_state = info_state

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        info_cards_pot = self._last_info_state.copy()[:54]
        info_street = PyPokerEngineStateExtractor.info_street(round_state = round_state)
        info_actions, legal_actions = PyPokerEngineStateExtractor.info_actions_legal_actions()
        info_state = np.concatenate((info_cards_pot, info_street, info_actions))
        
        reward = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] - self._round_initial_stack
        done = 1.0

        _ = self._nfsp_policy.step(info_state, legal_actions, reward, done, self._is_evaluation)
        self._last_info_state = None

