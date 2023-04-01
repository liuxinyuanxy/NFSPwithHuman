# from ai.base.fish_player import FishPlayer
# from ai.base.fold_man import FoldMan
from ai.base.honest_player import HonestPlayer
# from ai.base.random_player import RandomPlayer
from ai.base.sharp_player import SharpPlayer
from ai.base.weak_player import WeakPlayer
# AI player
from ai.nfsp.nfsp_player import NFSPPlayer
from ai.nfsp.fake_human_and_nfsp_player import FakeHumanAndNFSPPlayer
from ai.nfsp.fake_humen.WeakPlayerWithAI import WeakPlayerWithAI

from pypokerengine.api.game import setup_config, my_start_poker
import csv
import tqdm

from pypokerengine.engine.deck import Deck

results = {
    "nfsp_player vs honest_player": [],
}
players_name = ['fake_player', 'nsfp_player']
# complete results
for i in players_name:
    for j in players_name:
        if i != j:
            results[i + " vs " + j] = []
cheat_deck = Deck(cheat=True, cheat_card_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # start_hand, community_card

round = 100
bar = tqdm.tqdm(total = round * (len(players_name)**2 - len(players_name)) )
for _ in range(round):
    # weak_with_ai = WeakPlayerWithAI(
    #     config_dir="models/nfsp",
    #     name="nfsp_player",
    #     is_evaluation=True,
    #     device='cuda')

    fake_player = FakeHumanAndNFSPPlayer(
        config_dir="models/nfsp",
        name="nfsp_player1",
        is_evaluation=True,
        device='cuda')
    
    nfsp_player = NFSPPlayer(
        config_dir="models/nfsp",
        name="nfsp_player",
        is_evaluation=True,
        device='cuda')

    honest_player = HonestPlayer()
    # random_player = RandomPlayer()
    sharp_player = SharpPlayer()
    weak_player = WeakPlayer()
    # fish_player = FishPlayer()
    # fold_player = FoldMan()
    players = [fake_player, nfsp_player]
    
    for i in range(len(players)):
        for j in range(len(players)):
            if i != j:
                config = setup_config(max_round=5, initial_stack=2000, small_blind_amount=50)
                config.register_player(name=players_name[i], algorithm=players[i])
                config.register_player(name=players_name[j], algorithm=players[j])
                game_result = my_start_poker(config, initial_btn=1, verbose=0)
                result = game_result['players'][0]['stack'] - 2000
                results[players_name[i] + " vs " + players_name[j]].append(result)
                bar.update(1)

# save result to csv

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['player1', 'player2', 'result'])
    for key in results:
        for i in range(len(results[key])):
            writer.writerow([key.split(" vs ")[0], key.split(" vs ")[1], results[key][i]])
