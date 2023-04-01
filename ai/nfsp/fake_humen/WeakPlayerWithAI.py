from ai.nfsp.fake_human_and_nfsp_player import FakeHumanAndNFSPPlayer

class WeakPlayerWithAI(FakeHumanAndNFSPPlayer):
    def __init__(self, config_dir, name, is_evaluation, device):
        super().__init__(config_dir, name, is_evaluation, device)

    # 1. if ai raise a too high amount, human will raise no more than 1000
    # 2. if human think his win rate is too low, he will fallback to a weak action
    def human_declare_action(self, valid_actions, win_rate, ranked_actions):
        r0_act = ranked_actions[0]['action']
        r0_amt = ranked_actions[0]['amount']

        if win_rate < ( 1.0 / self.nb_player ):
            # if human think his win rate is too low, he will fallback to a weak action

            # if ai raise
            # human will fallback to a raise less than half
            # or call/fold
            if r0_act.startswith('r'):
                for item in ranked_actions:
                    if item['prob'] <= 0:
                        break
                    if item['action'] == 'raise':
                        if item['amount'] <= r0_amt/2:
                            return item['action'], item['amount']
                    else:
                        return item['action'], item['amount']
                return self.parse_action('call',valid_actions )

            # if ai call or fold
            # human will fold
            else:
                return self.parse_action('fold',valid_actions )
            
        # if win rate is not too low
        # if ai raise too high
        # human will raise no more than 1000
        elif r0_act.startswith('r') and r0_amt > 1000:
            return r0_act,1000.0
        

        return r0_act, r0_amt            
