from pypokerengine.api.game import setup_config, my_start_poker
# import ai
from ai.nfsp.nfsp_player import NFSPPlayer
import time
import tqdm

start_time = time.perf_counter()

ai1 = NFSPPlayer(config_dir="models/nfsp",
                 name="nfsp_player1",
                 is_evaluation=False,
                 device = 'cuda'
                #  device='cpu'
                 )
ai2 = NFSPPlayer(config_dir="models/nfsp",
                 name="nfsp_player2",
                 is_evaluation=False,
                 device = 'cuda'
                #  device='cpu'
                 )
config = setup_config(max_round=10, initial_stack=2000, small_blind_amount=50)
config.register_player(name="nfsp_player1",
                       algorithm=ai1
                       )
config.register_player(name="nfsp_player2",
                       algorithm=ai2
                       )
# episode = 30000
# episode = 100000
# episode = 5000000
pbar = tqdm.tqdm(total=episode)
for i in range(episode):
    game_result = my_start_poker(config, initial_btn=i % 2, verbose=0)
    pbar.update(1)

ai1.save()
ai2.save()
end_time = time.perf_counter()
print('Running time: %s Seconds' % (end_time - start_time))
