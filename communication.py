# system call: this_agent gameConfig server port
# gameConfig:
# server : localhost
# port : 10000
import socket
import sys
import uuid

from pypokerengine.players import BasePokerPlayer
from ai.nfsp.nfsp_player import NFSPPlayer

DEFAULT_SEAT = 1
config: dict = {}
round_num = 0
seat_num = 0


def main():
    aiplayer = NFSPPlayer(config_dir="models/nfsp",
                          name="player" + str(DEFAULT_SEAT),
                          is_evaluation=True,
                          device='cpu'
                          )
    # load config from file, file name comes from command line
    load_config(sys.argv[1])
    hostname = sys.argv[2]
    port = int(sys.argv[3])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((hostname, port))
    to_server = sock.makefile('w')
    from_server = sock.makefile('r')
    if to_server is None or from_server is None:
        print("ERROR: could not get socket streams", file=sys.stderr)
        return
    # send version string to dealer
    to_server.write("VERSION:2.0.0\n")
    to_server.flush()
    init_game()
    send_game_start_message(aiplayer)
    # play game
    while True:
        street_num: int = 0
        while True:
            state = from_server.readline()
            if not state:
                break
            state = state.strip()
            if state.startswith("MATCHSTATE"):
                break
        if not state:
            break
        _, _, hole_cards = parse_state(state)
        send_round_start_message(aiplayer, hole_cards)
        while True:
            state = from_server.readline()
            if not state:
                break
            state = state.strip()
            if state.startswith("MATCHSTATE"):
                action, board, hole_cards = parse_state(state)
        if not state:
            break


def load_config(filename: str):
    file = open(filename, "r")
    configs = file.readlines()
    file.close()
    conf = {}
    for line in configs:
        line = line.strip()
        if (len(line) < 2) or (line[0] == '#'):
            continue
        if line == "GAMEDEF":
            continue
        if line == "nolimit":
            conf["limit"] = False
            continue
        if line == "limit":
            conf["limit"] = True
            continue
        if line == "END GAMEDEF":
            break
        line = line.split("=")
        conf[line[0]] = line[1]
    global config
    config = conf
    init_game()
    return


def parse_valid_actions(config: dict):
    valid_actions = []
    if config["maxraises"] is not None:
        max_raises = int(config["maxraises"])
    else:
        max_raises = int(config["stack"].split(" ")[seat_num])
    min_raises = max([int(blind) for blind in config["blind"].split(" ")])
    valid_actions.append({"action": "fold", "amount": 0})
    valid_actions.append({"action": "call", "amount": 0})
    valid_actions.append({"action": "raise", "amount": {"max": max_raises, "min": min_raises}})
    return valid_actions


# MATCHSTATE:1:0::|9hQd|
# MATCHSTATE:1:1:ccr1642r16019cr20000c:|Qs5d|
def parse_state(state: str):
    global seat_num
    global round_num
    state = state.split(":")
    seat_num = int(state[1])
    round_num = int(state[2])
    action = state[3]
    cards = state[4]
    cards = cards.split("|")
    hole_cards = cards[config["__numplayers__"]]
    hole_cards = parse_cards(hole_cards)
    board = cards[-1]
    board.replace("/", "")
    board = parse_cards(board)
    return action, board, hole_cards


class RoundState:
    def __init__(self, round_count=1, dealer_btn=0, small_blind_pos=1, big_blind_pos=2, street='preflop', next_player=0,
                 community_card=None, seats=None, pot=None):
        if pot is None:
            pot = {}
        if seats is None:
            seats = []
        if community_card is None:
            community_card = []
        self.round_count = round_count
        self.dealer_btn = dealer_btn
        self.small_blind_pos = small_blind_pos
        self.big_blind_pos = big_blind_pos
        self.street = street
        self.next_player = next_player
        self.community_card = community_card
        self.seats = seats
        self.pot = pot

    def update(self, state: str):
        action, board, hole_cards = parse_state(state)
        if self.seats.__len__() == 0:
            self.seats = parse_seats()
            self.pot = {'main': {'amount': 0}, 'side': []}
        self.community_card = board


# convert 'Qs5d' to ['SQ', 'D5']
def parse_cards(cards: str):
    return [cards[i + 1].upper() + cards[i].upper() for i in range(0, len(cards), 2)]


def parse_hole_card(hole_cards: str):
    hole_cards = hole_cards.split("|")
    hole_cards = hole_cards[seat_num]
    return parse_cards(hole_cards)


def init_game():
    global config
    config["__numplayers__"] = int(config["numplayers"])
    uuids = [str(uuid.uuid4()) for _ in range(config["__numplayers__"])]
    name = [f'player{i}' for i in range(config["__numplayers__"])]
    stacks = [int(stack) for stack in config["stack"].split(" ")]
    config["__BB__"] = max([int(blind) for blind in config["blind"].split(" ")])
    config["__SB__"] = sum([int(blind) for blind in config["blind"].split(" ")]) - config["__BB__"]
    config["__uuids__"] = uuids
    config["__name__"] = name
    config["__stacks__"] = stacks
    return


def parse_seats():
    seats = []
    for i in range(config["__numplayers__"]):
        seat = {
            "name": config["__name__"][i],
            "stack": config["__stacks__"][i],
            "state": "participating",
            "uuid": config["__uuids__"][i]
        }
        seats.append(seat)
    return seats


def send_game_start_message(player: BasePokerPlayer):
    game_info = {
        'player_num': config["__numplayers__"],
        'rule': {
            'ante': 5,
            'blind_structure': {
                5: {"ante": 0, "smalll_blind": 0},
            },
            'max_round': config["numrounds"],
            'initial_stack': config["__stacks__"][0],
            'small_blind_amount': config["__SB__"],
        },
        'seats': parse_seats()
    }
    player.receive_game_start_message(game_info)


def send_round_start_message(player: BasePokerPlayer, hole_cards: list):
    player.receive_round_start_message(round_num, hole_cards, parse_seats())


if __name__ == "__main__":
    main()
