import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

plt.switch_backend('TkAgg') 

# GPU Device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
print(f"Current device is: {device}")

# Reading data from csv
df = pd.read_csv('./Data/drop_IBM.csv')
df['Date'] = pd.to_datetime(df['Date']) # Changing to datetime object
df['PriceChange'] = df['Close'] - df['Open'] # Price difference

# Data split
train_ratio = 0.5
val_ratio = 0.2
test_ratio = 0.3

total_size = len(df)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

train_data = df[:train_size]
val_data = df[train_size:train_size+val_size]
test_data = df[train_size+val_size:]

# Custom Stock Dataset
class StockDataset(Dataset):
    def __init__(self, df, context_length=30, initial_capital=10000):
        self.df = df
        self.context_length = context_length
        self.initial_capital = initial_capital
        self.current_capital = initial_capital  # 생성자에서 초기화
        self.num_stocks = 0
        self.data = self.preprocess_data()

    def preprocess_data(self):
        # Decision Transformer 입력 형태로 데이터 변환 (-1: sell, 0: hold, 1: buy)
        data = []
        # current_capital = self.initial_capital  # 초기화 제거
        for i in range(self.context_length, len(self.df)):
            context = self.df[i - self.context_length:i]
            returns_to_go = context['PriceChange'].sum()
            state = context[['Open', 'Close']].values.flatten()
            current_price = context['Close'].iloc[-1]

            # 현재 자본금을 고려하여 매수 가능 여부 결정
            if self.current_capital >= current_price:  # self.current_capital 사용
                action = np.sign(context['PriceChange'].iloc[-1])  # 원래 행동
                self.current_capital -= current_price
                self.num_stocks += 1
            else:
                action = 0  # 매수 불가능 시 Hold로 변경

            data.append((returns_to_go, state, action))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        returns_to_go, state, action = self.data[idx]
        action = torch.tensor(action, dtype=torch.float)
        return torch.tensor(returns_to_go, dtype=torch.float), torch.tensor(state, dtype=torch.float), action
    
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, context_length, n_layer=32, n_head=16, n_embd=512, dropout=0.3):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim * context_length, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.returns_embedding = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.positional_encoding = nn.Parameter(torch.zeros(1, context_length, n_embd))

        config = GPT2Config(
            vocab_size=1, 
            n_positions=context_length, 
            n_embd=n_embd, 
            n_layer=n_layer, 
            n_head=n_head
        )
        self.transformer = GPT2Model(config)

        self.action_predictor = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd, action_dim)
        )

    def forward(self, returns_to_go, states, actions):
        batch_size = states.size(0)

        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions.unsqueeze(-1))
        returns_embeddings = self.returns_embedding(returns_to_go.unsqueeze(-1))

        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
        stacked_inputs += self.positional_encoding[:, :stacked_inputs.size(1), :]

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        x = transformer_outputs.last_hidden_state[:, 0]
        action_logits = self.action_predictor(x)
        
        return action_logits
    
# Hyperparameter
context_length = 5
state_dim = 2  # Open, Close
action_dim = 3 # Sell, Hold, Buy
learning_rate = 0.01
batch_size = 16
epochs = 5

# 데이터 로더 생성 (Test, Val)
train_dataset = StockDataset(train_data, context_length)
val_dataset = StockDataset(val_data, context_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# GPU로 모델 전송
model = DecisionTransformer(state_dim, action_dim, context_length).to(device)

# 옵티마이저 및 손실 함수 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 기록을 위한 리스트
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 테스트 결과 기록을 위한 리스트
test_evaluation_values_list = [] 
buy_count = 0
sell_count = 0
hold_count = 0

# 초기 자본금 설정
initial_capital = 10000

# 학습 루프
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for returns_to_go, states, actions in train_loader:
        returns_to_go = returns_to_go.to(device)
        states = states.to(device)
        actions = actions.to(device)

        # 모델 출력 및 손실 계산
        action_logits = model(returns_to_go, states, actions)
        loss = criterion(action_logits, actions)

        # 정확도 계산
        _, predicted = torch.max(action_logits.data, 1)
        train_total += actions.size(0)
        train_correct += (predicted == actions).sum().item()

        # 옵티마이저 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 검증 손실 및 정확도 계산
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for returns_to_go, states, actions in val_loader:
            returns_to_go = returns_to_go.to(device)
            states = states.to(device)
            actions = actions.to(device)
            action_logits = model(returns_to_go, states, actions)
            loss = criterion(action_logits, actions)
            _, predicted = torch.max(action_logits.data, 1)
            val_total += actions.size(0)
            val_correct += (predicted == actions).sum().item()
            val_loss += loss.item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total

    # 지표 기록
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print('-' * 60)

    if (epoch + 1) % 1 == 0:
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        # 체크포인트 모델로 테스트 데이터셋 평가 및 시각화
        model.eval()
        with torch.no_grad():
            # 테스트 데이터셋 로더 생성 (추가)
            test_dataset = StockDataset(test_data, context_length)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            num_stocks = 0 # 보유 주식 수
            current_capital = initial_capital # 현재 자본금 (매 에폭마다 초기화)
            evaluation_values = [initial_capital] # 평가금 변화 기록
            for i, (returns_to_go, states, actions) in enumerate(test_loader):
                returns_to_go = returns_to_go.to(device)
                states = states.to(device)
                actions = actions.to(device)

                action_preds = model(returns_to_go, states, actions)
                action_preds = action_preds.view(-1, 3)
                predicted_actions = torch.argmax(action_preds, dim=1)

                # 예측된 행동에 따라 자본금 조정
                for j, predicted_action in enumerate(predicted_actions):
                    current_price = test_data['Close'].iloc[i * batch_size + j]
                    if predicted_action == 0:
                        # 보유
                        hold_count += 1
                        pass
                    elif predicted_action == 1:
                        # 매수
                        if current_capital >= current_price:  # 자본금이 충분한 경우에만 매수
                            buy_count += 1
                            num_stocks += 1
                            current_capital -= current_price 
                        else:
                            print(f"Step: {i * batch_size + j}, 자본금 부족으로 매수 불가") 
                    elif predicted_action == 2:
                        # 매도
                        sell_count += 1
                        num_stocks -= 1
                        current_capital += current_price
                    
                    evaluation_value = current_capital + num_stocks * current_price  # 평가금 계산
                    evaluation_values.append(evaluation_value)

            test_evaluation_values_list.append(evaluation_values)

# 그래프 그리기
for i, evaluation_values in enumerate(test_evaluation_values_list):
    plt.plot(evaluation_values, label=f'Checkpoint {i * 5 + 5}')
plt.xlabel('Step')
plt.ylabel('Evaluation Value')
plt.legend()
plt.title('Evaluation Value Change over Test Dataset (Checkpoints)')

# 최종 그래프
plt.plot(test_evaluation_values_list[-1], label=f'Final Epoch')
plt.xlabel('Step')
plt.ylabel('Evaluation Value')
plt.legend()
plt.title('Evaluation Value Change (Final Epoch)')
plt.tight_layout()
plt.show()

# 최종 이익 계산 및 출력
final_evaluation_value = test_evaluation_values_list[-1][-1]
profit_percentage = ((final_evaluation_value - initial_capital) / initial_capital) * 100
print(f"최종 이익: {final_evaluation_value:.2f}")
print(f"초기 자본금 대비 이익률: {profit_percentage:.2f}%")

# Buy, Sell, Hold 횟수 출력
print(f"Buy 횟수: {buy_count}")
print(f"Sell 횟수: {sell_count}")
print(f"Hold 횟수: {hold_count}")