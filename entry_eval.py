from pypokerengine.api.game import setup_config, my_start_poker
from ai.nfsp.nfsp_player import NFSPPlayer

ai1 = NFSPPlayer(config_dir="models/nfsp",
                 name="nfsp_player1",
                 is_evaluation=True,
                 device='cpu'
                 )
ai2 = NFSPPlayer(config_dir="models/nfsp",
                 name="nfsp_player2",
                 is_evaluation=True,
                 device='cpu'
                 )
config = setup_config(max_round=1, initial_stack=20000, small_blind_amount=50)
config.register_player(name="nfsp_player1",
                       algorithm=ai1
                       )
config.register_player(name="nfsp_player2",
                       algorithm=ai2
                       )
episode = 1000
for i in range(episode):
    game_result = my_start_poker(config, initial_btn=i % 2, verbose=1)
