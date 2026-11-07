from sklearn.decomposition import TruncatedSVD

def compute_svd(ppmi_mat, dim=100):
    print(f"Computing Truncated SVD (dim={dim})...")
    svd = TruncatedSVD(n_components=dim, random_state=42)
    embeddings = svd.fit_transform(ppmi_mat)
    return embeddings