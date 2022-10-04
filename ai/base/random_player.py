import random as rand

from pypokerengine.players import BasePokerPlayer


class RandomPlayer(BasePokerPlayer):

    def __init__(self):
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0 / 3

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    def declare_action(self, valid_actions, hole_card, round_state):
        choice = self.__choice_action(valid_actions)
        print(valid_actions)
        print(choice)
        action = choice["action"]
        amount = choice["amount"]
        if action == "raise":
            amount = rand.randrange(amount["min"], max(amount["min"], amount["max"]) + 1)
        return action, amount

    def __choice_action(self, valid_actions):
        r = rand.random()
        if r <= self.fold_ratio:
            ttuple = self.parse_action("fold", valid_actions)
            return {"action": ttuple[0], "amount": ttuple[1]}
        elif r <= self.call_ratio:
            ttuple = self.parse_action("call", valid_actions)
            return {"action": ttuple[0], "amount": ttuple[1]}
        else:
            ttuple = self.parse_action("raise", valid_actions)
            return {"action": ttuple[0], "amount": ttuple[1]}

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
