import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr
from probe_models import StructuralDistanceProbe, StructuralDepthProbe

def train_distance_probe(train_data, layer_idx, rank=64, epochs=30, lr=1e-3, batch_size=20, device="cpu"):
    probe = StructuralDistanceProbe(rank=rank).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    for epoch in range(epochs):
        probe.train()
        permutation = torch.randperm(len(train_data))
        for i in range(0, len(train_data), batch_size):
            batch_indices = permutation[i:i + batch_size]
            loss = 0.0
            for idx in batch_indices:
                h = train_data[idx]["layer_embeddings"][layer_idx].to(device)
                target = train_data[idx]["dist_matrix"].to(device)
                pred = probe(h)
                loss += torch.mean(torch.abs(pred - target))
            
            optimizer.zero_grad()
            (loss / len(batch_indices)).backward()
            optimizer.step()
    return probe

def evaluate_distance_probe(probe, test_data, layer_idx, device="cpu"):
    probe.eval()
    total_correct_edges, total_gold_edges = 0, 0
    with torch.no_grad():
        for item in test_data:
            h = item["layer_embeddings"][layer_idx].to(device)
            gold_dist = item["dist_matrix"].numpy()
            n = gold_dist.shape[0]
            
            pred_dist = probe(h).cpu().numpy()
            # MST decoding
            mst = minimum_spanning_tree(pred_dist).toarray()
            
            # Gold edges have tree distance == 1
            pred_edges = {(min(i, j), max(i, j)) for i in range(n) for j in range(n) if mst[i, j] > 0 or mst[j, i] > 0}
            gold_edges = {(min(i, j), max(i, j)) for i in range(n) for j in range(i + 1, n) if gold_dist[i, j] == 1.0}
            
            total_correct_edges += len(pred_edges & gold_edges)
            total_gold_edges += len(gold_edges)
            
    return total_correct_edges / total_gold_edges if total_gold_edges > 0 else 0.0

def train_depth_probe(train_data, layer_idx, rank=64, epochs=30, lr=1e-3, batch_size=20, device="cpu"):
    probe = StructuralDepthProbe(rank=rank).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    
    for epoch in range(epochs):
        probe.train()
        permutation = torch.randperm(len(train_data))
        for i in range(0, len(train_data), batch_size):
            batch_indices = permutation[i:i + batch_size]
            loss = 0.0
            for idx in batch_indices:
                h = train_data[idx]["layer_embeddings"][layer_idx].to(device)
                target = train_data[idx]["depth_vector"].to(device)
                pred = probe(h)
                loss += torch.mean(torch.abs(pred - target))
                
            optimizer.zero_grad()
            (loss / len(batch_indices)).backward()
            optimizer.step()
    return probe

def evaluate_depth_probe(probe, test_data, layer_idx, device="cpu"):
    probe.eval()
    spearman_scores = []
    with torch.no_grad():
        for item in test_data:
            h = item["layer_embeddings"][layer_idx].to(device)
            target = item["depth_vector"].numpy()
            pred = probe(h).cpu().numpy()
            if len(target) > 1:
                corr, _ = spearmanr(pred, target)
                if not torch.isnan(torch.tensor(corr)):
                    spearman_scores.append(corr)
    return sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0.0