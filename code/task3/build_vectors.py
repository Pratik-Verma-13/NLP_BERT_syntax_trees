import torch
import conllu
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def build_bert_static_vectors(conllu_path, layer_idx=8, model_name="bert-base-uncased", device="cpu"):
    """Extracts contextual embeddings from a specific layer and averages them into static vectors."""
    print(f"Parsing {conllu_path}...")
    with open(conllu_path, "r", encoding="utf-8") as f:
        data = conllu.parse(f.read())

    print(f"Loading {model_name} on {device}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    word_sums = defaultdict(lambda: torch.zeros(768, device=device))
    word_counts = defaultdict(int)

    with torch.no_grad():
        for sentence in tqdm(data, desc=f"Averaging Layer {layer_idx} vectors"):
            valid_tokens = [tok for tok in sentence if isinstance(tok["id"], int)]
            if not valid_tokens:
                continue
                
            # Lowercase every word to match vocabulary casing requirements
            words = [tok["form"].lower() for tok in valid_tokens]
            
            subword_indices = []
            token_ids = [tokenizer.cls_token_id]
            for word in words:
                sub_tokens = tokenizer.encode(word, add_special_tokens=False)
                if not sub_tokens:
                    sub_tokens = [tokenizer.unk_token_id]
                subword_indices.append(len(token_ids))
                token_ids.extend(sub_tokens)
            token_ids.append(tokenizer.sep_token_id)
            
            if len(token_ids) > 512:
                continue

            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            outputs = model(input_tensor)
            
            # Extract the specific optimal layer
            layer_rep = outputs.hidden_states[layer_idx][0, subword_indices, :]
            
            for i, word in enumerate(words):
                word_sums[word] += layer_rep[i]
                word_counts[word] += 1

    print(f"Consolidating vocabulary of size {len(word_counts)}...")
    static_vectors = {}
    for word in word_counts:
        static_vectors[word] = (word_sums[word] / word_counts[word]).cpu()
        
    return static_vectors