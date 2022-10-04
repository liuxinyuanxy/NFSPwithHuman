from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random

NB_SIMULATION = 100


class SharpPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=NB_SIMULATION,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )
        rn = random.randint(1, 10)
        if win_rate >= 1.0 / self.nb_player:
            return self.parse_action("call", valid_actions)  # fetch CALL action info
        else:
            if rn >= 4:
                return self.parse_action("call", valid_actions)
            else:
                return self.parse_action("fold", valid_actions)  # fetch FOLD action info

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
