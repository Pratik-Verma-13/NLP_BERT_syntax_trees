from collections import defaultdict
import scipy.sparse as sp

def build_cooccurrence_matrix(sentences, word2idx, window_size=2):
    print(f"Building co-occurrence matrix (window={window_size})...")
    vocab_size = len(word2idx)
    data_dict = defaultdict(float)
    
    for sent in sentences:
        indices = [word2idx[w] for w in sent if w in word2idx]
        for i, target in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context = indices[j]
                    data_dict[(target, context)] += 1.0
                    
    row, col, data = [], [], []
    for (t, c), count in data_dict.items():
        row.append(t)
        col.append(c)
        data.append(count)
        
    return sp.csr_matrix((data, (row, col)), shape=(vocab_size, vocab_size))