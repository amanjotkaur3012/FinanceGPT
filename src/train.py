import torch
import os
import sys
import glob
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# Automatically find the project root and add it to system path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import FinanceGPT 

# --- ENTERPRISE CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8       
BLOCK_SIZE = 128     
EPOCHS = 40          
LEARNING_RATE = 5e-4
USE_AMP = torch.cuda.is_available() 

print(f"🚀 Launching Optimized Training on: {DEVICE.upper()}")

# 1. LOAD TOKENIZER (Using absolute paths)
token_path = os.path.join(BASE_DIR, "models", "tokenizer.json")
if not os.path.exists(token_path):
    print("❌ ERROR: Tokenizer not found! Run src/build_tokenizer.py first.")
    exit()

tokenizer = Tokenizer.from_file(token_path)
vocab_size = tokenizer.get_vocab_size()

# 2. PREPARE DATASET
data_files = glob.glob(os.path.join(BASE_DIR, "data", "*.txt"))
raw_text = ""
for file in data_files:
    with open(file, "r", encoding="utf-8") as f:
        raw_text += f.read() + "\n\n"

class FinanceDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokens = tokenizer.encode(text).ids
        self.block_size = block_size
    def __len__(self):
        return len(self.tokens) - self.block_size - 1
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

dataset = FinanceDataset(raw_text, tokenizer, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. INITIALIZE MODEL & SCALER
model = FinanceGPT(vocab=vocab_size, dim=256, layers=4, heads=4, max_len=BLOCK_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

# 4. ADVANCED TRAINING LOOP
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                logits, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        
    print(f"📈 Epoch {epoch+1:02d}/{EPOCHS} | Perplexity: {torch.exp(torch.tensor(total_loss/len(dataloader))):.2f} | Loss: {total_loss/len(dataloader):.4f}")

save_path = os.path.join(BASE_DIR, "models", "final_model.pt")
torch.save(model.state_dict(), save_path)
print(f"🎉 Enterprise Model Weights Saved to {save_path}!")