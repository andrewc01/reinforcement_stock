def calculate_reward(profit_ratio, target_profit_ratio, mdd, principal):
  """
  Reward Function 계산

  Args:
    profit_ratio: 현재 수익률
    target_profit_ratio: 목표 수익률
    mdd: 최대 손실 (Maximum Drawdown)
    principal: 투자 원금

  Returns:
    reward: 계산된 보상 값
  """
  reward = 0

  # 목표 수익률 달성에 대한 보상
  if profit_ratio >= target_profit_ratio:
    reward += principal * (profit_ratio - target_profit_ratio) * 10
  else:
    reward += principal * (profit_ratio / target_profit_ratio)

  # 위험 관리에 대한 보상
  reward -= mdd * principal

  return reward



