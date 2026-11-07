import os
import torch
import random
import numpy as np
from build_vectors import build_bert_static_vectors
from eval_vectors import evaluate_wordsim353, evaluate_google_analogy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = "cpu"
    print(f"Running Task 3 on {device}...")

    conllu_path = "data/en_gum-ud-train.conllu"
    wordsim_path = "data/wordsim353.csv"
    analogy_path = "data/questions-words.txt"
    cache_path = "data/gum_static_vectors_layer8.pt"
    
    # Use the best syntactic layer identified in Task 2
    optimal_layer = 8

    # Step 1: Build or Load Static Vectors
    if os.path.exists(cache_path):
        print(f"\nLoading cached static vectors from {cache_path}...")
        static_vectors = torch.load(cache_path)
    else:
        print(f"\nBuilding static vectors from {conllu_path} (Layer {optimal_layer})...")
        static_vectors = build_bert_static_vectors(
            conllu_path, layer_idx=optimal_layer, device=device
        )
        torch.save(static_vectors, cache_path)
        print(f"Cached {len(static_vectors)} word vectors.")

    # Step 2: Evaluate WordSim-353
    print("\n--- Evaluating WordSim-353 ---")
    ws_corr, ws_oov, ws_total = evaluate_wordsim353(static_vectors, wordsim_path)
    print(f"Total Pairs: {ws_total}")
    print(f"OOV Dropped: {ws_oov}")
    print(f"Spearman Correlation: {ws_corr:.4f}")

    # Step 3: Evaluate Google Analogy Set
    print("\n--- Evaluating Google Analogy Set ---")
    an_acc, an_oov, an_total = evaluate_google_analogy(static_vectors, analogy_path)
    print(f"Total Questions: {an_total}")
    print(f"OOV Dropped: {an_oov} (Expected to be high for a small treebank vocab)")
    print(f"Accuracy (on in-vocab questions): {an_acc * 100:.2f}%")

if __name__ == "__main__":
    main()