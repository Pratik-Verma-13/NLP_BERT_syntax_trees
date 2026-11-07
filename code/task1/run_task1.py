import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from dataset import parse_conllu, build_vocabulary
from cooccurrence import build_cooccurrence_matrix
from ppmi import compute_ppmi
from svd_embeddings import compute_svd
from evaluation import evaluate_wordsim353, evaluate_google_analogy

def main():
    train_path = "data/en_gum-ud-train.conllu"
    wordsim_path = "data/wordsim353.csv"
    analogy_path = "data/questions-words.txt"
    
    print("\n=== Starting Task 1: Classical Distributional Semantics ===")
    
    # 1. Dataset & Vocab
    sentences = parse_conllu(train_path)
    word2idx, vocab = build_vocabulary(sentences, min_freq=2)
    
    # 2. Co-occurrence Matrix
    cooc_mat = build_cooccurrence_matrix(sentences, word2idx, window_size=2)
    
    # 3. PPMI Matrix
    ppmi_mat = compute_ppmi(cooc_mat)
    
    # 4. Truncated SVD 
    embeddings = compute_svd(ppmi_mat, dim=100)
    
    # 5. Evaluation
    evaluate_wordsim353(embeddings, word2idx, wordsim_path)
    evaluate_google_analogy(embeddings, word2idx, vocab, analogy_path)
    
    print("\n=== Task 1 Completed Successfully! ===")

if __name__ == "__main__":
    main()