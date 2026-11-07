import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

def evaluate_wordsim353(vectors, wordsim_path):
    """Computes Spearman correlation for human similarity judgments."""
    gold_scores = []
    pred_scores = []
    oov_count = 0
    total_pairs = 0
    
    with open(wordsim_path, "r", encoding="utf-8") as f:
        for line in f:
            # Replace commas with spaces to handle both CSV and TSV formats
            parts = line.replace(",", " ").split()
            
            if len(parts) < 3: 
                continue
                
            w1, w2 = parts[0].lower(), parts[1].lower()
            
            # Safely try to parse the score; this automatically skips the header row
            try:
                score = float(parts[2])
            except ValueError:
                continue
                
            total_pairs += 1
            
            if w1 in vectors and w2 in vectors:
                v1, v2 = vectors[w1].unsqueeze(0), vectors[w2].unsqueeze(0)
                sim = F.cosine_similarity(v1, v2).item()
                pred_scores.append(sim)
                gold_scores.append(score)
            else:
                oov_count += 1
                
    evaluated = total_pairs - oov_count
    corr = spearmanr(pred_scores, gold_scores).correlation if evaluated > 1 else 0.0
    return corr, oov_count, total_pairs


def evaluate_google_analogy(vectors, analogy_path):
    """Evaluates analogy questions: a is to b as c is to d -> max_d cos(d, b - a + c)."""
    vocab = list(vectors.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    print("Building normalized embedding matrix for analogy task...")
    embed_matrix = torch.stack([vectors[w] for w in vocab])
    embed_matrix = F.normalize(embed_matrix, p=2, dim=1)
    
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
            
            # target = b - a + c
            target_vec = embed_matrix[idx_b] - embed_matrix[idx_a] + embed_matrix[idx_c]
            target_vec = F.normalize(target_vec, p=2, dim=0).unsqueeze(0)
            
            # Vectorized cosine similarity
            sims = torch.matmul(target_vec, embed_matrix.t()).squeeze(0)
            
            # Mask out the input words so they aren't predicted as the answer
            sims[idx_a] = -float('inf')
            sims[idx_b] = -float('inf')
            sims[idx_c] = -float('inf')
            
            pred_idx = torch.argmax(sims).item()
            if vocab[pred_idx] == expected_d:
                correct += 1
                
    evaluated = total_questions - oov_count
    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    return accuracy, oov_count, total_questions