import torch
from torch import nn, optim
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_count):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, head_count, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x_norm = self.norm1(x)
        attention_output, _ = self.attention(x_norm,x_norm,x_norm)
        x = x + self.dropout(attention_output)
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        return x

class Transformer_learnable(nn.Module):
    def __init__(self, vocab_size, posi_size, embed_size, num_layers, head_count, LB):
        super(Transformer_learnable, self).__init__()
        self.embed_size = embed_size
        self.LB = LB

        # embed each z_bit ∈ {0,1}
        self.z_bit_embed = nn.Embedding(2, embed_size)
        # project flattened z_emb → one context token
        self.z_ffn = nn.Sequential(
            nn.Linear(LB * embed_size, embed_size),
            nn.ReLU()
        )

        # embedding for P/b tokens
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(posi_size, embed_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, head_count) for _ in range(num_layers)]
        )

        # predict next b ∈ {0,1}
        self.fc_out = nn.Linear(embed_size, 2)
        self.tokendrop = nn.Dropout(0.25)

    def forward(self, inputs, mask=None):
        """
        inputs: LongTensor of shape (B, LB + T)
        """
        device = inputs.device
        B, total_len = inputs.size()

        # 1) embed z_bits
        z_bits = inputs[:, :self.LB].long()               # (B, LB)
        z_emb  = self.z_bit_embed(z_bits.to(device))      # (B, LB, D)
        z_flat = z_emb.flatten(start_dim=1)                # (B, LB*D)
        para_context = self.z_ffn(z_flat).unsqueeze(1)    # (B, 1, D)

        # 2) embed interleaved tokens
        token_inputs = inputs[:, self.LB:].long()         # (B, T)
        token_emb    = self.word_embedding(token_inputs.to(device))  # (B, T, D)
        positions    = torch.arange(token_inputs.size(1), device=device) \
                               .unsqueeze(0).expand(B, -1)         # (B, T)
        pos_emb      = self.position_embedding(positions)          # (B, T, D)
        token_emb    = token_emb + pos_emb                       # (B, T, D)
        token_emb    = self.tokendrop(token_emb)                  # 如果要用 dropout

        # 3) concat
        out = torch.cat((para_context, token_emb), dim=1)         # (B, 1+T, D)

        # 4) transformer layers
        for layer in self.layers:
            out = layer(out)

        # 5) predict next b
        logits = self.fc_out(out[:, -1, :])                       # (B, 2)
        return logits


def generate_with_z(z_bits, model, LA, sample_size, device):
    """
    Args:
        z_bits: 1D LongTensor of shape (LB,)  e.g. torch.tensor([0,1,0,1,1,0,0,1])
        model: your Transformer_learnable instance
        LA: number of subsystem A qubits (e.g. 2)
        sample_size: how many sequences to generate
        device: torch.device
    Returns:
        outputs: LongTensor of shape (sample_size, LB + 2*LA)
                 each row = [z1..z8, P1, b1, P2, b2]
    """
    LB = z_bits.numel()
    T = 2 * LA
    # 1) 初始化 outputs tensor
    outputs = torch.zeros((sample_size, LB + T), dtype=torch.long, device=device)
    # 2) 填入相同的 z_bits
    z_expand = z_bits.to(device).unsqueeze(0).expand(sample_size, LB)
    outputs[:, :LB] = z_expand

    # 3) 逐位生成 (P_i, b_i)
    for i in range(LA):
        P_col = LB + 2 * i
        b_col = P_col + 1

        # 隨機選一個 P ∈ {2,3,4}
        outputs[:, P_col] = torch.randint(2, 5, (sample_size,), device=device)

        # 呼叫 model，傳入到目前為止的 prefix
        logits = model(outputs[:, :P_col+1], device)  # shape (sample_size, 2)
        probs = F.softmax(logits, dim=1)

        # 從 Bernoulli(pred_b=1 機率) 取樣
        b_sample = torch.bernoulli(probs[:, 1]).to(torch.long)
        outputs[:, b_col] = b_sample

    return outputs

QB = 13

# 1) 設定裝置和超參數
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
print(f"Using device: {device}")

# 2) 用 pandas 讀 CSV，然後轉成 LongTensor
data_path = f"U1_dataset_LA=2_LB=14_QB={QB}.csv"
df = pd.read_csv(data_path)                # 讀出 DataFrame
sequences = torch.tensor(df.values, dtype=torch.long)

# 3) 拆分 train / val
N         = len(sequences)
train_len = int(0.8 * N)
val_len   = N - train_len
train_ds, val_ds = random_split(sequences, [train_len, val_len])

# 4) 建 DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

def train_epoch(model, loader, optim, crit, LB, LA, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for seq in loader:
        seq = seq.to(device)
        optim.zero_grad()
        loss = 0
        correct = 0
        for i in range(LA):
            P_idx, b_idx = LB+2*i, LB+2*i+1
            logits = model(seq[:, :P_idx+1])
            target = seq[:, b_idx]
            loss += crit(logits, target)
            pred = logits.argmax(1)
            correct += (pred == target).sum()
        loss = loss / LA          # 平均
        loss.backward()
        optim.step()

        total_loss += loss.item()*seq.size(0)
        total_correct += correct.item()
        total_samples += seq.size(0)*LA
    return total_loss/len(loader.dataset), total_correct/total_samples

def eval_epoch(model, loader, criterion, LB, LA, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for seq in loader:                          # seq shape (B, LB+2*LA)
            seq = seq.to(device)

            batch_loss, batch_correct = 0.0, 0
            for i in range(LA):
                P_idx, b_idx = LB + 2*i, LB + 2*i + 1
                logits  = model(seq[:, :P_idx+1])   # (B,2)
                target  = seq[:, b_idx]             # (B,)

                batch_loss += criterion(logits, target)
                pred = logits.argmax(1)
                batch_correct += (pred == target).sum()

            # 平均到每個 b
            batch_loss = batch_loss / LA

            # 乘上一個 batch 的樣本數，方便最後除以總資料量
            B = seq.size(0)
            total_loss    += batch_loss.item() * B
            total_correct += batch_correct.item()
            total_samples += B * LA

    avg_loss = total_loss / len(loader.dataset)        # 每個 b 的 CE
    accuracy = total_correct / total_samples           # 每個 b 的準確率
    return avg_loss, accuracy

# 超參數
vocab_size = 5     # P∈{2,3,4}, b∈{0,1}
posi_size  = 4     # 2*LA = 4
embed_size = 128
num_layers = 4
head_count = 4
LB = 14             # z₁…z₈
LA = 2             # P1,b1,P2,b2
epochs = 80
patience = 5      # 如果連續多少 epoch 驗證 loss 沒改善就 early stop

# 裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型、優化器、損失
model     = Transformer_learnable(vocab_size, posi_size, embed_size, num_layers, head_count, LB).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
no_improve    = 0

for epoch in range(1, epochs+1):
    print(f"\nEpoch {epoch}/{epochs}\n" + "-"*30)
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, LB, LA, device)
    val_loss,   val_acc   = eval_epoch (model, val_loader,   criterion, LB, LA, device)
    print(f"Epoch {epoch}: "
      f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
      f"val loss={val_loss:.4f} acc={val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'best_model_QB={QB}.pth')
        print("Saved Best Model")
    else:
        no_improve += 1
        print(f"No improvement for {no_improve}/{patience} epochs")
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
            break

# 保存最終模型（若在 early stop 前沒保存，或想另存最後狀態）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, f'final_model_QB={QB}.pth')