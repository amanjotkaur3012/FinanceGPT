import os
import glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

print("🚀 Initializing Custom Finance Tokenizer...")

# Automatically find the absolute path to your project folders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure the models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Find your text files
data_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

if not data_files:
    print(f"❌ ERROR: No .txt files found in {DATA_DIR}. Please check your folders.")
    exit()

print(f"📚 Reading {len(data_files)} datasets...")

# Configure and train the Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=3000, 
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

tokenizer.train(data_files, trainer)

# Save it to the exact absolute path
save_path = os.path.join(MODEL_DIR, "tokenizer.json")
tokenizer.save(save_path)

print(f"✅ Tokenizer built and saved to: {save_path}")
print(f"📊 Vocabulary Size: {tokenizer.get_vocab_size()} tokens")