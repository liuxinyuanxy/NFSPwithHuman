class BasePokerPlayer(object):
    """Base Poker client implementation

    To create poker client, you need to override this class and
    implement following 7 methods.

    - declare_action
    - receive_game_start_message
    - receive_round_start_message
    - receive_street_start_message
    - receive_game_update_message
    - receive_round_result_message
    """

    def __init__(self):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        err_msg = self.__build_err_msg("declare_action")
        raise NotImplementedError(err_msg)

    def receive_game_start_message(self, game_info):
        err_msg = self.__build_err_msg("receive_game_start_message")
        raise NotImplementedError(err_msg)

    def receive_round_start_message(self, round_count, hole_card, seats):
        err_msg = self.__build_err_msg("receive_round_start_message")
        raise NotImplementedError(err_msg)

    def receive_street_start_message(self, street, round_state):
        err_msg = self.__build_err_msg("receive_street_start_message")
        raise NotImplementedError(err_msg)

    def receive_game_update_message(self, new_action, round_state):
        err_msg = self.__build_err_msg("receive_game_update_message")
        raise NotImplementedError(err_msg)

    def receive_round_result_message(self, winners, hand_info, round_state):
        err_msg = self.__build_err_msg("receive_round_result_message")
        raise NotImplementedError(err_msg)

    def set_uuid(self, uuid):
        self.uuid = uuid

    def respond_to_ask(self, message):
        """Called from Dealer when ask message received from RoundManager"""
        valid_actions, hole_card, round_state = self.__parse_ask_message(message)
        return self.declare_action(valid_actions, hole_card, round_state)

    def receive_notification(self, message):
        """Called from Dealer when notification received from RoundManager"""
        msg_type = message["message_type"]

        if msg_type == "game_start_message":
            info = self.__parse_game_start_message(message)
            self.receive_game_start_message(info)

        elif msg_type == "round_start_message":
            round_count, hole, round_state = self.__parse_round_start_message(message)
            self.receive_round_start_message(round_count, hole, round_state)

        elif msg_type == "street_start_message":
            street, state = self.__parse_street_start_message(message)
            self.receive_street_start_message(street, state)

        elif msg_type == "game_update_message":
            new_action, round_state = self.__parse_game_update_message(message)
            self.receive_game_update_message(new_action, round_state)

        elif msg_type == "round_result_message":
            winners, hand_info, state = self.__parse_round_result_message(message)
            self.receive_round_result_message(winners, hand_info, state)

    def __build_err_msg(self, msg):
        return "Your client does not implement [ {0} ] method".format(msg)

    def __parse_ask_message(self, message):
        hole_card = message["hole_card"]
        valid_actions = message["valid_actions"]
        round_state = message["round_state"]
        return valid_actions, hole_card, round_state

    def __parse_game_start_message(self, message):
        game_info = message["game_information"]
        return game_info

    def __parse_round_start_message(self, message):
        round_count = message["round_count"]
        hole_card = message["hole_card"]
        round_state = message["round_state"]
        return round_count, hole_card, round_state

    def __parse_street_start_message(self, message):
        street = message["street"]
        round_state = message["round_state"]
        return street, round_state

    def __parse_game_update_message(self, message):
        new_action = message["action"]
        round_state = message["round_state"]
        return new_action, round_state

    def __parse_round_result_message(self, message):
        winners = message["winners"]
        hand_info = message["hand_info"]
        round_state = message["round_state"]
        return winners, hand_info, round_state

    def parse_action(self, action, valid_actions):
        for valid_action in valid_actions:
            if valid_action["action"] == action:
                return valid_action["action"], valid_action["amount"]
        if action == "call":
            return "fold", 0
        return "call", 0
