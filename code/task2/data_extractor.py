import os
import torch
import conllu
import networkx as nx
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def parse_conllu_dataset(file_path, max_length=50):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Treebank file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = conllu.parse(f.read())

    processed = []
    
    for sentence in data:
        words = []
        g = nx.Graph()
        root_idx = None
        
        # Keep only standard integer-indexed tokens (skip multiword range tokens e.g. 1-2)
        valid_tokens = [tok for tok in sentence if isinstance(tok["id"], int)]
        
        if len(valid_tokens) <= 1 or len(valid_tokens) > max_length:
            continue
            
        for tok in valid_tokens:
            idx = tok["id"] - 1
            words.append(tok["form"])
            g.add_node(idx)
            
            head_idx = tok["head"]
            if head_idx == 0:
                root_idx = idx
            elif head_idx is not None:
                g.add_edge(idx, head_idx - 1)
                
        # Validate that the dependency structure is a single connected tree
        if root_idx is None or not nx.is_connected(g) or g.number_of_edges() != (len(words) - 1):
            continue

        n = len(words)
        
        # 1. Compute all-pairs shortest paths (Gold Tree Distances)
        dist_matrix = torch.zeros((n, n), dtype=torch.float32)
        lengths = dict(nx.all_pairs_shortest_path_length(g))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = float(lengths[i][j])
                
        # 2. Compute depth from the syntactic root
        depth_vector = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            depth_vector[i] = float(lengths[root_idx][i])
            
        processed.append({
            "words": words,
            "dist_matrix": dist_matrix,
            "depth_vector": depth_vector
        })
        
    print(f"Successfully loaded and validated {len(processed)} sentences from {file_path}.")
    return processed


def extract_and_align_bert_features(parsed_data, model_name="bert-base-uncased", device="cpu"):

    print(f"Loading {model_name} for feature extraction on {device}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    dataset_features = []

    with torch.no_grad():
        for item in tqdm(parsed_data, desc="Extracting BERT layers"):
            words = item["words"]
            subword_indices = []
            
            # Subword alignment: Build token sequence and keep track of first subword indices
            token_ids = [tokenizer.cls_token_id]
            for word in words:
                sub_tokens = tokenizer.encode(word, add_special_tokens=False)
                if not sub_tokens:
                    sub_tokens = [tokenizer.unk_token_id]
                
                # The index of the FIRST subword in the full sequence
                subword_indices.append(len(token_ids))
                token_ids.extend(sub_tokens)
                
            token_ids.append(tokenizer.sep_token_id)
            
            if len(token_ids) > 512:
                continue

            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            outputs = model(input_tensor)
            
            aligned_layers = []
            for l in range(1, 13):
                layer_rep = outputs.hidden_states[l][0, subword_indices, :].detach().cpu()
                aligned_layers.append(layer_rep)

            dataset_features.append({
                "layer_embeddings": aligned_layers,  # List of 12 tensors of shape (N, 768)
                "dist_matrix": item["dist_matrix"],  # (N, N)
                "depth_vector": item["depth_vector"], # (N,)
                "length": len(words)
            })

    return dataset_features


def cache_extracted_dataset(features, output_path):
    """Caches pre-extracted representations to disk to save compute."""
    torch.save(features, output_path)
    print(f"Cached extracted features to {output_path}")


def load_cached_dataset(input_path):
    """Loads pre-extracted representations from disk."""
    print(f"Loading cached features from {input_path}...")
    return torch.load(input_path)


if __name__ == "__main__":
    # Smoke-test run
    sample_file = "data/en_ewt-ud-test.conllu"
    if os.path.exists(sample_file):
        parsed = parse_conllu_dataset(sample_file, max_length=20)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features = extract_and_align_bert_features(parsed[:5], device=device)
        print("Sample sentence 0 word count:", features[0]["length"])
        print("Sample sentence 0 Layer 1 tensor shape:", features[0]["layer_embeddings"][0].shape)
        print("Sample sentence 0 Distance matrix shape:", features[0]["dist_matrix"].shape)
    else:
        print(f"Test file {sample_file} not found. Place your assigned UD split in data/ to test.")