import numpy as np
from numpy import ndarray

import pypokerengine.utils.visualize_utils as U
from ai.nfsp.nfsp_player import PyPokerEngineStateExtractor, NFSPPlayer


class HumanAndNFSPPlayer(NFSPPlayer):
    def __init__(self,
                 config_dir,
                 name,
                 is_evaluation,
                 device,
                 input_receiver=None
                 ):
        super().__init__(config_dir, name, is_evaluation, device)
        self.input_receiver = input_receiver if input_receiver else self.__gen_raw_input_wrapper()

    def declare_action(self, valid_actions, hole_card, round_state):
        print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
        print('Below is what the bot said: ')
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
        for action_code, p in ranked_actions:
            if p <= 0:
                break
            if action_code == self._num_actions - 2:
                action = 'fold'
                amount = 0
            elif action_code == self._num_actions - 1:
                action = 'call'
                amount = [act['amount'] for act in valid_actions if act['action'] == action][0]
            elif action_code >= 0 and action_code < self._num_actions - 2:
                action = 'raise'
                amount = PyPokerEngineStateExtractor.parse_raiseby(info_state, action_code) \
                         + max(
                    [0.0] + [action['amount'] for action in round_state['action_histories'][round_state['street']] if
                             action['uuid'] == self.uuid])
            else:
                raise "error action"
            print(f"action:{action}, amount:{amount}, probability:{p}")
        self._last_info_state = info_state
        action, amount = self.__receive_action_from_console(valid_actions)
        return action, amount

    def receive_game_start_message(self, game_info):
        print(U.visualize_game_start(game_info, self.uuid))
        self.game_info = game_info
        self.__wait_until_input()

    def receive_round_start_message(self, round_count, hole_card, round_state):
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
        print(U.visualize_round_start(round_count, hole_card, round_state['seats'], self.uuid))
        self.__wait_until_input()

    def receive_street_start_message(self, street, round_state):
        print(U.visualize_street_start(street, round_state, self.uuid))
        self.__wait_until_input()

    def receive_game_update_message(self, action, round_state):
        print(U.visualize_game_update(action, round_state, self.uuid))
        self.__wait_until_input()

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
        print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
        self.__wait_until_input()

    def __wait_until_input(self):
        input("Enter some key to continue ...")

    def __gen_raw_input_wrapper(self):
        return lambda msg: input(msg)

    def __receive_action_from_console(self, valid_actions):
        valid_actions_dic, choices = self.__valid_actions_to_dic(valid_actions)

        flg = self.input_receiver(f'Enter {", ".join(choices)}.\n >> ')

        if flg in valid_actions_dic.keys():

            if flg == 'f':
                return valid_actions_dic['f']['action'], valid_actions_dic['f']['amount']
            elif flg == 'c':
                return valid_actions_dic['c']['action'], valid_actions_dic['c']['amount']
            elif flg == 'r':
                valid_amounts = valid_actions_dic['r']['amount']
                raise_amount = self.__receive_raise_amount_from_console(valid_amounts['min'], valid_amounts['max'])
                return valid_actions_dic['r']['action'], raise_amount
        else:
            return self.__receive_action_from_console(valid_actions)

    def __valid_actions_to_dic(self, valid_actions):
        actions_dic = {}
        choices = []
        for action in valid_actions:
            if action['action'].startswith('f'):
                choices.append('f(fold)')
                actions_dic['f'] = action
            elif action['action'].startswith('c'):
                choices.append('c(call)')
                actions_dic['c'] = action
            elif action['action'].startswith('r'):
                choices.append('r(raise)')
                actions_dic['r'] = action

        return actions_dic, choices

    def __receive_raise_amount_from_console(self, min_amount, max_amount):
        raw_amount = self.input_receiver("valid raise range = [%d, %d]\n" % (min_amount, max_amount))
        try:
            amount = int(raw_amount)
            if min_amount <= amount and amount <= max_amount:
                return amount
            else:
                print("Invalid raise amount %d. Try again.")
                return self.__receive_raise_amount_from_console(min_amount, max_amount)
        except:
            print("Invalid input received. Try again.")
            return self.__receive_raise_amount_from_console(min_amount, max_amount)
