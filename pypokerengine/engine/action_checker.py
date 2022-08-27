from functools import reduce

class ActionChecker:

  @classmethod
  def correct_action(self, players, player_pos, sb_amount, action, amount=None):
    if self.is_allin(players[player_pos], action, amount):
      amount = players[player_pos].stack + players[player_pos].paid_sum()
    elif self.__is_illegal(players, player_pos, sb_amount, action, amount):
      action, amount = "fold", 0
    return action, amount


  @classmethod
  def is_allin(self, player, action, bet_amount):
    if action == 'call':
      return bet_amount >= player.stack + player.paid_sum()
    elif action == 'raise':
      return bet_amount == player.stack + player.paid_sum()
    else:
      return False


  @classmethod
  def need_amount_for_action(self, player, amount):
    return amount - player.paid_sum()


  @classmethod
  def agree_amount(self, players):
    last_raise = self.__fetch_last_raise(players)
    return last_raise["amount"] if last_raise else 0


  # def __fetch_last_raise(self, players):
  #   all_histories = [p.action_histories for p in players]
  #   all_histories = reduce(lambda acc, e: acc + e, all_histories)  # flatten
  #   raise_histories = [h for h in all_histories if h["action"] in ["RAISE", "SMALLBLIND", "BIGBLIND"]]
  #   if len(raise_histories) == 0:
  #     return None
  #   else:
  #     return max(raise_histories, key=lambda h: h["amount"])  # maxby

  @classmethod
  def legal_actions(self, players, player_pos, sb_amount):
    '''
    如果存在raise字段且min>=max 则表示只能all in
    '''
    actions = []

    agree_amount = self.agree_amount(players)
    paid_sum = players[player_pos].paid_sum()
    if agree_amount > paid_sum:
            actions.append({ "action" : "fold" , "amount" : 0 })

    actions.append({ "action" : "call" , "amount" : agree_amount })


    max_raise = players[player_pos].stack + players[player_pos].paid_sum()

    if max_raise > agree_amount:
      acted = (len([action for action in players[player_pos].action_histories if action['action'] in ['RAISE', 'CALL']]) > 0)

      min_raise = self.__min_raise_amount(players, sb_amount)
      min_raise_add_amount = min_raise-agree_amount      

      if acted:
        reopen = ((agree_amount-paid_sum)>min_raise_add_amount)

      if not acted or reopen:
        min_raise = agree_amount + min_raise_add_amount
        actions.append( { "action" : "raise", "amount" : { "min": min_raise, "max": max_raise }} )

    return actions





  @classmethod
  def _is_legal(self, players, player_pos, sb_amount, action, amount=None):
    return not self.__is_illegal(players, player_pos, sb_amount, action, amount)

  @classmethod
  def __is_illegal(self, players, player_pos, sb_amount, action, amount=None):
    if action == 'fold':
      return False
    elif action == 'call':
      return self.__is_short_of_money(players[player_pos], amount)\
          or self.__is_illegal_call(players, amount)
    elif action == 'raise':
      return self.__is_short_of_money(players[player_pos], amount) \
          or self.__is_illegal_raise(players, player_pos, amount, sb_amount)

  @classmethod
  def __is_illegal_call(self, players, amount):
    return amount != self.agree_amount(players)

  @classmethod
  def __is_illegal_raise(self, players, player_pos, amount, sb_amount):
    player = players[player_pos]
    min_raise = self.__min_raise_amount(players, sb_amount)
    max_raise = player.paid_sum() + player.stack
    if min_raise <= max_raise:
      return amount < min_raise or amount > max_raise
    else:
      return amount != max_raise

  @classmethod
  def __min_raise_amount(self, players, sb_amount):
    raises = self.__fetch_certain_action_histories(players, ['RAISE'])
    raises_add_amount = [h['add_amount'] for h in raises]
    min_raise_add_amount = max(raises_add_amount+[sb_amount*2])
    agree_amount = self.agree_amount(players)
    return agree_amount + min_raise_add_amount

  @classmethod
  def __is_short_of_money(self, player, amount):
    return player.stack < amount - player.paid_sum()

  @classmethod
  def __fetch_last_raise(self, players):
    raise_histories = self.__fetch_certain_action_histories(players, ["RAISE", "SMALLBLIND", "BIGBLIND"])
    if len(raise_histories) == 0:
      return None
    else:
      return max(raise_histories, key=lambda h: h["amount"])  # maxby

  @classmethod
  def __fetch_certain_action_histories(self, players, action_types):
    all_histories = [p.action_histories for p in players]
    all_histories = reduce(lambda acc, e: acc + e, all_histories)  # flatten
    certain_action_histories = [h for h in all_histories if h["action"] in action_types]
    return certain_action_histories

