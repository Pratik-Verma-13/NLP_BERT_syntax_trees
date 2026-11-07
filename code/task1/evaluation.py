import numpy as np
from scipy.stats import spearmanr

def evaluate_wordsim353(embeddings, word2idx, wordsim_path):
    print("\n--- Evaluating WordSim-353 ---")
    gold_scores, pred_scores = [], []
    oov_count, total_pairs = 0, 0
    
    with open(wordsim_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            w1, w2 = parts[0].lower(), parts[1].lower()
            try:
                score = float(parts[2])
            except ValueError:
                continue
                
            total_pairs += 1
            if w1 in word2idx and w2 in word2idx:
                v1 = embeddings[word2idx[w1]]
                v2 = embeddings[word2idx[w2]]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                pred_scores.append(sim)
                gold_scores.append(score)
            else:
                oov_count += 1
                
    evaluated = total_pairs - oov_count
    corr = spearmanr(pred_scores, gold_scores).correlation if evaluated > 1 else 0.0
    print(f"Total Pairs: {total_pairs}")
    print(f"OOV Dropped: {oov_count}")
    print(f"Spearman Correlation: {corr:.4f}")

def evaluate_google_analogy(embeddings, word2idx, vocab, analogy_path):
    print("\n--- Evaluating Google Analogy Set ---")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed_embeds = embeddings / norms
    
    oov_count, correct, total_questions = 0, 0, 0
    
    with open(analogy_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(":"):
                continue
            parts = line.strip().lower().split()
            if len(parts) != 4:
                continue
                
            total_questions += 1
            a, b, c, expected_d = parts
            
            if not all(w in word2idx for w in (a, b, c, expected_d)):
                oov_count += 1
                continue
                
            idx_a, idx_b, idx_c = word2idx[a], word2idx[b], word2idx[c]
            
            target_vec = normed_embeds[idx_b] - normed_embeds[idx_a] + normed_embeds[idx_c]
            target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)
            
            sims = np.dot(normed_embeds, target_vec)
            sims[idx_a] = -np.inf
            sims[idx_b] = -np.inf
            sims[idx_c] = -np.inf
            
            pred_idx = np.argmax(sims)
            if vocab[pred_idx] == expected_d:
                correct += 1
                
    evaluated = total_questions - oov_count
    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    print(f"Total Questions: {total_questions}")
    print(f"OOV Dropped: {oov_count}")
    print(f"Accuracy (on in-vocab): {accuracy * 100:.2f}%")