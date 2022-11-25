from ai.base.fish_player import FishPlayer
from ai.base.fold_man import FoldMan
from ai.base.honest_player import HonestPlayer
from ai.base.random_player import RandomPlayer
from ai.base.sharp_player import SharpPlayer
from ai.base.weak_player import WeakPlayer
from ai.nfsp.nfsp_player import NFSPPlayer
from pypokerengine.api.game import setup_config, my_start_poker
import csv

results = {
    "nfsp_player vs honest_player": [],
}
players_name = ['nfsp_player', 'honest_player', 'random_player', 'sharp_player', 'weak_player', 'fish_player',
                'fold_player']
# complete results
for i in players_name:
    for j in players_name:
        if i != j:
            results[i + " vs " + j] = []

for _ in range(10):
    nfsp_player = NFSPPlayer(config_dir="models/nfsp",
                             name="nfsp_player1",
                             is_evaluation=True,
                             device='cpu'
                             )
    honest_player = HonestPlayer()
    random_player = RandomPlayer()
    sharp_player = SharpPlayer()
    weak_player = WeakPlayer()
    fish_player = FishPlayer()
    fold_player = FoldMan()
    players = [nfsp_player, honest_player, random_player, sharp_player, weak_player, fish_player, fold_player]
    for i in range(len(players)):
        for j in range(len(players)):
            if i != j:
                config = setup_config(max_round=5, initial_stack=20000, small_blind_amount=50)
                config.register_player(name=players_name[i], algorithm=players[i])
                config.register_player(name=players_name[j], algorithm=players[j])
                game_result = my_start_poker(config, initial_btn=1, verbose=1)
                result = game_result['players'][0]['stack'] - 2000
                results[players_name[i] + " vs " + players_name[j]].append(result)

# save result to csv

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['player1', 'player2', 'result'])
    for key in results:
        for i in range(len(results[key])):
            writer.writerow([key.split(" vs ")[0], key.split(" vs ")[1], results[key][i]])
