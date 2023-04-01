import numpy as np
from numpy import ndarray

import pypokerengine.utils.visualize_utils as U
from ai.nfsp.nfsp_player import PyPokerEngineStateExtractor, NFSPPlayer
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

NB_SIMULATION = 1000


class FakeHumanAndNFSPPlayer(NFSPPlayer):
    def __init__(self,
                 config_dir,
                 name,
                 is_evaluation,
                 device
                 ):
        super().__init__(config_dir, name, is_evaluation, device)
        self._round_initial_stack = None
        self.nb_player = None
        self.game_info = None
        self._last_info_state = None

    # ranked_actions: [{'action': 'fold/call/raise', 'amount': float,}]
    def human_declare_action(self, valid_actions, win_rate, ranked_actions):
        if win_rate < 1.0 / self.nb_player:
            if ranked_actions[0]['action'].startswith('r'):
                return self.parse_action('call',valid_actions)
            else:
                return self.parse_action('fold',valid_actions )
        return ranked_actions[0]['action'], ranked_actions[0]['amount']

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        
        # human decision
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )
        info_cards = PyPokerEngineStateExtractor.info_cards(round_state['community_card'] + hole_card)
        info_pot = PyPokerEngineStateExtractor.info_pot(round_state=round_state, agent=self)
        info_street = PyPokerEngineStateExtractor.info_street(round_state=round_state)
        info_actions, legal_actions = PyPokerEngineStateExtractor.info_actions_legal_actions(
            valid_actions=valid_actions, round_state=round_state, agent=self)
        info_state = np.concatenate((info_cards, info_pot, info_street, info_actions))
        reward = 0.0
        done = 0.0
        actions: ndarray = self._nfsp_policy.step(info_state, legal_actions, reward, done, self._is_evaluation,
                                                  is_human=True)
        sorted_indices = np.argsort(actions)
        sorted_indices = sorted_indices[::-1]

        ranked_actions = [[i, actions[i]] for i in sorted_indices]
        action_: str = ''
        amount = 0
        ranked_actions_readable = []
        for action_code, p in ranked_actions:
            if p <= 0:
                break
            if action_code == self._num_actions - 2:
                if win_rate < 1.0 / self.nb_player:
                    action_ = 'call'
                    amount = [act['amount'] for act in valid_actions if act['action'] == action_][0]
                else:
                    action_ = 'fold'
                    amount = 0
            elif action_code == self._num_actions - 1:
                if win_rate < 1.0 / self.nb_player and len([act['action'] for act in valid_actions if act['action'] == "raise"]) > 0:
                    action_ = 'raise'
                    amount = PyPokerEngineStateExtractor.parse_raiseby(info_state, 0) \
                            + max(
                        [0.0] + [float(action['amount']) for action in round_state['action_histories'][round_state['street']] if
                                action['uuid'] == self.uuid])
                else:
                    action_ = 'call'
                    amount = [act['amount'] for act in valid_actions if act['action'] == action_][0]
            elif 0 <= action_code < self._num_actions - 2:
                action_ = 'raise'
                amount = PyPokerEngineStateExtractor.parse_raiseby(info_state, action_code) \
                         + max(
                    [0.0] + [float(action['amount']) for action in round_state['action_histories'][round_state['street']] if
                             action['uuid'] == self.uuid])
            else:
                raise "error action"
            break
        self._last_info_state = info_state

        # return action_, amount
        return self.human_declare_action(valid_actions, win_rate, ranked_actions_readable)
        

    def receive_game_start_message(self, game_info):
        self.game_info = game_info

    def receive_round_start_message(self, round_count, hole_card, round_state):
        self.nb_player = len(round_state['seats'])
        stack = sum([player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid],
                    0.0)  # 使用sum防止本金不够不通知，这个list顶多一个元素
        ante_blind = sum([action['amount'] for action in round_state['action_histories'].get('preflop', []) if
                          action['uuid'] == self.uuid], 0.0)
        self._round_initial_stack = stack + ante_blind
        info_cards = PyPokerEngineStateExtractor.info_cards(hole_card)
        info_pot = PyPokerEngineStateExtractor.info_pot(round_state=round_state, agent=self)
        info_street = PyPokerEngineStateExtractor.info_street(round_state=round_state)
        info_actions, _ = PyPokerEngineStateExtractor.info_actions_legal_actions()
        info_state = np.concatenate((info_cards, info_pot, info_street, info_actions))
        self._last_info_state = info_state

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        info_cards_pot = self._last_info_state.copy()[:54]
        info_street = PyPokerEngineStateExtractor.info_street(round_state=round_state)
        info_actions, legal_actions = PyPokerEngineStateExtractor.info_actions_legal_actions()
        info_state = np.concatenate((info_cards_pot, info_street, info_actions))

        reward = [player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][
                     0] - self._round_initial_stack
        done = 1.0

        _ = self._nfsp_policy.step(info_state, legal_actions, reward, done, self._is_evaluation)
        self._last_info_state = None


