import torch
import os
import sys
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import FinanceGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_ai():
    print("Loading AI Model into memory...")
    token_path = os.path.join(BASE_DIR, "models", "tokenizer.json")
    model_path = os.path.join(BASE_DIR, "models", "final_model.pt")
    
    tokenizer = Tokenizer.from_file(token_path)
    vocab_size = tokenizer.get_vocab_size()
    
    model = FinanceGPT(vocab=vocab_size, dim=256, layers=4, heads=4, max_len=128)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    return model, tokenizer

def generate_answer(model, tokenizer, question, context, max_tokens=50):
    # THE FIX: Match the prompt exactly to how the model was trained (flat text)
    prompt = f"{context} "
    
    ids = tokenizer.encode(prompt).ids
    prompt_length = len(ids)
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        out = model.generate(x, steps=max_tokens, top_k=3)[0].tolist()

    reply = tokenizer.decode(out[prompt_length:])
    
    # Make it look professional by ending at a full sentence
    if "." in reply:
        reply = reply.split(".")[0] + "."
    
    return reply.strip()