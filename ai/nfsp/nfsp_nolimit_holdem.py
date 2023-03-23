import enum
import abc
import numpy as np
import torch
import torch.nn.functional as F
import sys

class Action(enum.Enum):
        RAISE_0 = 0
        RAISE_1 = 1
        RAISE_2 = 2
        RAISE_3 = 3
        RAISE_4 = 4
        RAISE_5 = 5
        RAISE_6 = 6
        RAISE_7 = 7
        RAISE_8 = 8
        RAISE_9 = 9
        RAISE_10 = 10
        RAISE_11 = 11
        RAISE_12 = 12
        RAISE_13 = 13
        RAISE_14 = 14
        RAISE_15 = 15
        RAISE_16 = 16
        RAISE_17 = 17
        RAISE_18 = 18
        RAISE_19 = 19

        FOLD = 20
        CALL = 21

NUM_INFO_STATE = 54 + 5 + 5

STREET_PARSE = {
    "preflop" : 0,
    "flop" : 1,
    "turn" : 2,
    "river" : 3,
    "showdown":4
}

class StateExtractor(metaclass = abc.ABCMeta):
    '''
    提取状态信息
        info_state(float) info_cards 52位0-1手牌信息，
                            info_pot 2位下注信息(self, pot)， 
                            info_street 5位street信息，
                            info_actions num_action位合法动作信息
                                fold
                                call byamount
                                can raise
                                minraise by 
                                all in by
        legal_action(bool) num_action位合法动作信息
        reward(float) 回报
        done(bool) 是否终局
    '''
    STATE_REPRESENTATION_SIZE = 54 + 5 + 5
    NUM_ACTIONS = len(Action)
    CARD_TO_INDEX = {'As': 39, '2s': 40, '3s': 41, '4s': 42, '5s': 43, '6s': 44, '7s': 45, '8s': 46, '9s': 47, 'Ts': 48, 'Js': 49, 'Qs': 50, 'Ks': 51, 'Ah': 26, '2h': 27, '3h': 28, '4h': 29, '5h': 30, '6h': 31, '7h': 32, '8h': 33, '9h': 34, 'Th': 35, 'Jh': 36, 'Qh': 37, 'Kh': 38, 'Ad': 13, '2d': 14, '3d': 15, '4d': 16, '5d': 17, '6d': 18, '7d': 19, '8d': 20, '9d': 21, 'Td': 22, 'Jd': 23, 'Qd': 24, 'Kd': 25, 'Ac': 0, '2c': 1, '3c': 2, '4c': 3, '5c': 4, '6c': 5, '7c': 6, '8c': 7, '9c': 8, 'Tc': 9, 'Jc': 10, 'Qc': 11, 'Kc': 12}
    @classmethod
    @abc.abstractmethod
    def _card2index(cls, card):
        '''
        将card转成0-51的id
        '''
        pass

    @classmethod
    def info_cards(cls, cards):
        info_cards = np.zeros(52, dtype = float)
        idx = np.array([cls._card2index(card) for card in cards])
        info_cards[idx] = 1.0
        return info_cards

    @classmethod
    @abc.abstractmethod
    def _get_mychip_pot(cls, **kwargs):
        pass

    @classmethod
    def info_pot(cls, **kwargs):
        mychip, pot = cls._get_mychip_pot(**kwargs)

        return np.array([
                            mychip,
                            pot
                        ],
                        dtype = float)

    @classmethod
    @abc.abstractmethod
    def _get_street(cls, **kwargs):
        pass

    @classmethod
    def info_street(cls, **kwargs):
        street = cls._get_street(**kwargs)
        info_street = np.zeros(5, dtype = float)
        idx = STREET_PARSE.get(street)
        if not idx is None:
            info_street[idx] = 1.0
        return info_street

    @classmethod
    @abc.abstractmethod
    def _get_valid_action(cls, **kwargs):
        pass

    @classmethod
    def info_actions_legal_actions(cls, **kwargs):
        num_actions = len(Action)
        info_actions = np.zeros(5, dtype = float)
        legal_actions = np.zeros(num_actions, dtype = bool)

        valid_actions, stack= cls._get_valid_action(**kwargs)
        # {
        #     "fold": True
        #     "call": bycall
        #     "raise":min by raise
        # }

        if valid_actions:
            fold = valid_actions.get('fold', None)
            if fold != None:
                info_actions[0] = 1.0
                legal_actions[num_actions - 2] = True

            bycall = valid_actions.get('call', None)
            if bycall != None:
                info_actions[1] = bycall
                legal_actions[num_actions - 1] = True

            min_by_raise = valid_actions.get('raise', None)
            if min_by_raise != None:
                info_actions[2] = 1.0
                info_actions[3] = min_by_raise
                if min_by_raise < stack:
                    legal_actions[0:num_actions-2] = True
                else:
                    legal_actions[-3] = True

            info_actions[4] = stack

        return info_actions, legal_actions

    @classmethod
    def get_info_actions(cls, info_state):
        return info_state[-5:]

    @classmethod
    def parse_raiseby(cls, info_state, action_code):
        info_actions = cls.get_info_actions(info_state)
        num_actions = len(Action)
        if info_actions[2] != 1 or action_code < 0 or action_code >= num_actions - 2:
            raise 'cant raise'

        #return round(info_actions[3]+(info_actions[4]-info_actions[3])*((action_code/(num_actions-2-1))**2))
        return round(info_actions[3]+(info_actions[4]-info_actions[3])*((action_code/(num_actions-2-1))))


def _byvalue(info_state):
    num_actions = len(Action)
    byvalue = torch.Tensor(num_actions).float().to(device = info_state.device)
    action_codes = torch.arange(num_actions).to(device = info_state.device)
    byvalue[:num_actions-2] = info_state[62]+(info_state[63]-info_state[62])*((action_codes[:num_actions-2]/(num_actions-2-1)))
    #byvalue[:num_actions-2] = info_state[62]+(info_state[63]-info_state[62])*((action_codes[:num_actions-2]/(num_actions-2-1))**2)
    byvalue[num_actions-2] = -info_state[52]#fold by (-mypot)
    byvalue[num_actions-1] = info_state[60]#call by
    return byvalue

def _to_legal_actions(info_state):
    num_actions = len(Action)
    legal_actions = torch.ones(num_actions).bool().to(device = info_state.device)
    if info_state[59] < 1:
        legal_actions[num_actions-2] = False

    if info_state[61] < 1:
        legal_actions[:num_actions-2] = False

    else:
        if info_state[62]>info_state[63]:
            legal_actions[:num_actions-3] = False
    return legal_actions

def sl_loss_backward(info_states, outputs, actions):
    device = info_states.device
    batch_size = info_states.size()[0]
    num_actions = len(Action)

    actions = actions.unsqueeze(dim = 1)

    byvalues = torch.ones([batch_size, num_actions]).float().to(device = device)
    legal_actions = torch.ones([batch_size, num_actions]).bool().to(device = device)
    for i in range(batch_size):
        byvalues[i] = _byvalue(info_states[i])
        legal_actions[i] = _to_legal_actions(info_states[i])


    rough_probs = F.softmax(outputs, dim = 1)
    ignored_predict = torch.ones([batch_size, num_actions]).float().to(device = device)
    ignored_predict[~legal_actions] = rough_probs[~legal_actions]
    loss1 = F.mse_loss(torch.sum(byvalues*ignored_predict, dim = 1)/100, torch.zeros(batch_size).float().to(device = device))

    label = torch.gather(input = byvalues, dim = 1, index = actions)/100#让raise 范围缩减至[2,200]

    pruned_outputs = torch.ones([batch_size, num_actions]).float().to(device = device)*(-sys.maxsize)
    pruned_outputs[legal_actions] = outputs[legal_actions]
    probs = F.softmax(pruned_outputs, dim = 1)

    predict = torch.sum(byvalues*probs, dim = 1)/100#

    loss2 = F.mse_loss(predict, label.squeeze(1))

    loss = loss1 + loss2
    loss.backward()

    return loss


