import torch
import numpy as np
import random
from data_extractor import parse_conllu_dataset, extract_and_align_bert_features
from train_evaluate import train_distance_probe, evaluate_distance_probe, train_depth_probe, evaluate_depth_probe
from plot_results import plot_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = "cpu"
    
    train_path = "data/en_gum-ud-train.conllu"
    test_path = "data/en_gum-ud-test.conllu"
    
    # Step 1: Parsing
    print("\n--- Step 1: Parsing UD English GUM ---")
    train_parsed = parse_conllu_dataset(train_path)
    test_parsed = parse_conllu_dataset(test_path)

    # If full GUM is too large for your system's RAM:
    if len(train_parsed) > 2500:
        print(f"Subsampling 2,500 sentences from {len(train_parsed)} to fit local memory...")
        train_parsed = train_parsed[:2500]

    # Step 2: Extraction
    print("\n--- Step 2: Extracting frozen BERT embeddings ---")
    train_features = extract_and_align_bert_features(train_parsed, device=device)
    test_features = extract_and_align_bert_features(test_parsed, device=device)

    ranks = [32, 64, 128]
    results = {r: {"uuas": [], "depth_spearman": []} for r in ranks}
    
    best_layer = None
    best_uuas = -1.0
    
    print("3. Sweeping ranks and layers...")
    for r in ranks:
        print(f"\n--- Testing Rank k = {r} ---")
        for layer in range(12):  # 0 to 11 (corresponds to layers 1 to 12)
            dist_probe = train_distance_probe(train_features, layer, rank=r, epochs=30, batch_size=20, device=device)
            uuas = evaluate_distance_probe(dist_probe, test_features, layer, device=device)
            results[r]["uuas"].append(uuas)

            depth_probe = train_depth_probe(train_features, layer, rank=r, epochs=30, batch_size=20, device=device)
            spearman = evaluate_depth_probe(depth_probe, test_features, layer, device=device)
            results[r]["depth_spearman"].append(spearman)
            
            print(f"Layer {layer + 1:2d} | UUAS: {uuas:.4f} | Spearman: {spearman:.4f}")
            
            if uuas > best_uuas:
                best_uuas = uuas
                best_layer = layer + 1
                
    print(f"\nBest syntactic layer identified: Layer {best_layer} (l*) with UUAS: {best_uuas:.4f}")
    plot_metrics(results, ranks=ranks)

if __name__ == "__main__":
    main()