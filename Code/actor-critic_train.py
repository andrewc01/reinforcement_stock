import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda') # nvidia gpu
    elif torch.backends.mps.is_available():
        return torch.device('mps') # apple silicon gpu
    else:
        return torch.device('cpu') # default
    
device = get_default_device()

# 데이터 전처리
df = pd.read_csv('./Data/drop_aapl.csv') # 주가 데이터
data = df['Close'].values  # 주가 데이터 추출 (우선 마감 가격만)
data = (data - data.mean()) / data.std()  # 스케일링

# train, test 데이터 분리 (8:2 비율)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

seq_len = 30  # 시퀀스 길이 (결과에 따라 조절)
# train 데이터 전처리
X_train, y_train = [], []
for i in range(len(train_data) - seq_len):
    X_train.append(train_data[i:i+seq_len])
    y_train.append(train_data[i+seq_len])
X_train, y_train = np.array(X_train), np.array(y_train)

# test 데이터 전처리
X_test, y_test = [], []
for i in range(len(test_data) - seq_len):
    X_test.append(test_data[i:i+seq_len])
    y_test.append(test_data[i+seq_len])
X_test, y_test = np.array(X_test), np.array(y_test)

# 환경 설정
class StockTradingEnv:
    def __init__(self, data, init_capital=10000, transaction_cost=0.001):
        self.data = np.array(data)
        self.init_capital = init_capital
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.t = 0
        self.capital = self.init_capital
        self.shares = 0
        self.state = self.data[:seq_len].tolist()  # 초기 상태는 30일 주가 시퀀스
        return self.state

    def step(self, action):
        # 행동 해석: 0 = 매도, 1 = 홀드, 2 = 매수
        price = self.data[self.t + seq_len]  # 다음 날짜의 주가
        if action == 0:  # 매도
            self.capital += self.shares * price * (1 - self.transaction_cost)
            self.shares = 0
        elif action == 2:  # 매수
            shares_to_buy = int(self.capital / price)
            self.shares += shares_to_buy
            self.capital -= shares_to_buy * price * (1 + self.transaction_cost)

        self.t += 1
        done = self.t >= len(self.data) - seq_len
        next_state = self.data[self.t:self.t + seq_len].tolist()
        reward = self.capital + self.shares * price - self.init_capital
        info = {}

        return next_state, reward, done, info
    
# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 출력은 2차원 (batch_size, hidden_size)이므로 squeeze(0)으로 1차원으로 변환
        _, (h_n, _) = self.lstm(x)
        action_probs = self.fc(h_n.squeeze(0))  # squeeze(0)으로 batch 차원 제거
        action_probs = nn.functional.softmax(action_probs, dim=0)  # softmax 적용
        return action_probs
    
# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        value = self.fc(h_n.squeeze(0))
        return value

  
# Actor-Critic 에이전트
class ActorCriticAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.policy_net = PolicyNetwork(input_size, hidden_size, output_size).to(device)  # GPU로 이동
        self.value_net = ValueNetwork(input_size, hidden_size).to(device)  # GPU로 이동
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters())
        self.value_optimizer = optim.AdamW(self.value_net.parameters())

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # GPU로 이동
        action_probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict['actions']).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).to(device)

        # Critic 업데이트
        values = self.value_net(states).squeeze(-1)
        next_values = self.value_net(next_states).squeeze(-1)
        expected_values = rewards + (1 - dones) * 0.99 * next_values
        value_loss = torch.mean((values - expected_values) ** 2)

        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True) 

        for param in self.value_net.parameters():
            param.data = param.data - self.value_optimizer.param_groups[0]['lr'] * param.grad.data 
        self.value_optimizer.zero_grad()

        # Actor 업데이트
        action_probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        advantages = expected_values - values
        policy_loss = -torch.mean(log_probs * advantages)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()  # retain_graph=True 필요 없음
        
        for param in self.policy_net.parameters():
            param.data = param.data - self.policy_optimizer.param_groups[0]['lr'] * param.grad.data
        self.policy_optimizer.zero_grad()

# 하이퍼파라미터 설정
num_episodes = 5000
early_stopping_patience = 100

# In-place 연산 감지
torch.autograd.set_detect_anomaly(True) 

# 학습 및 평가
env = StockTradingEnv(train_data)
input_size = seq_len  # 시퀀스 길이
hidden_size = 64
output_size = 3  # 매수, 매도, 홀드
agent = ActorCriticAgent(input_size, hidden_size, output_size) 

best_reward = -np.inf
early_stopping_counter = 0
episode_rewards = [] # 에피소드별 보상 저장

# 학습 루프
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(reward)
        transition_dict['next_states'].append(next_state)
        transition_dict['dones'].append(done)

        state = next_state
        episode_reward += reward

    agent.update(transition_dict)
    episode_rewards.append(episode_reward)
    print(f'Episode: {episode}, Reward: {episode_reward}')

    # 조기 종료 조건 확인
    if episode_reward > best_reward:
        best_reward = episode_reward
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at episode {episode}')
            break

# 평가
test_env = StockTradingEnv(test_data, init_capital=10000) # test 데이터로 평가
state = test_env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done, _ = test_env.step(action)
    total_reward += reward
    state = next_state

print(f'Test Reward: {total_reward}')

# 그래프 그리기
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards')
plt.show()