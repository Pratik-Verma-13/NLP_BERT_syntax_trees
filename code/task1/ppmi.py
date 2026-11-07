import numpy as np
import scipy.sparse as sp

def compute_ppmi(cooc_mat):
    print("Computing PPMI matrix...")
    total_sum = cooc_mat.sum()
    row_sums = np.array(cooc_mat.sum(axis=1)).flatten()
    col_sums = np.array(cooc_mat.sum(axis=0)).flatten()
    
    cooc_coo = cooc_mat.tocoo()
    expected = (row_sums[cooc_coo.row] * col_sums[cooc_coo.col]) / total_sum
    
    pmi = np.zeros_like(cooc_coo.data)
    valid = (cooc_coo.data > 0) & (expected > 0)
    pmi[valid] = np.log2(cooc_coo.data[valid] / expected[valid])
    
    ppmi = np.maximum(0, pmi)
    mask = ppmi > 0
    
    return sp.csr_matrix((ppmi[mask], (cooc_coo.row[mask], cooc_coo.col[mask])), shape=cooc_mat.shape)