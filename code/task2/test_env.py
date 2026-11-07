import torch
from transformers import BertTokenizer, BertModel
import conllu
import scipy
import networkx as nx

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Test BERT initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

sample_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sample_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # 13 hidden states: layer 0 (embeddings) + layers 1-12
    hidden_states = outputs.hidden_states

print(f"BERT loaded successfully. Extracted {len(hidden_states) - 1} transformer layers.")
print(f"Layer 1 output shape: {hidden_states[1].shape} (batch, seq_len, 768)")