import torch
import numpy as np

def generate_domain_data(domain_idx, num_samples=250, d=768, out_dim=10, seed=42):
    """
    Generates synthetic high-dimensional data representing 30 heterogeneous domains.
    - Domains 1-8: Phase 1 (Finance & Numbers) -> Sparse, high-variance noise
    - Domains 9-15: Phase 2 (Law & Logic) -> Binary logical relational structures
    - Domains 16-22: Phase 3 (NLP & General) -> Dense clustered distributions
    - Domains 23-30: Phase 4 (Vision-Flattened) -> Spatial grid sinusoids & block features
    """
    # Fix seeds for deterministic generation per domain
    torch.manual_seed(seed + domain_idx)
    np.random.seed(seed + domain_idx)

    # Determine Phase
    if 1 <= domain_idx <= 8:
        # Phase 1: Finance & Numbers (Sparse, high-variance features)
        # We sample features from a diagonal covariance with highly unequal scale
        scales = torch.exp(torch.randn(d) * 1.5)
        raw_X = torch.randn(num_samples, d) * scales
        # Sparsify (keep only 10% non-zero features)
        mask = (torch.rand(num_samples, d) < 0.1).float()
        X = raw_X * mask
    elif 9 <= domain_idx <= 15:
        # Phase 2: Law & Logic (Binary-like logic features)
        # Generate discrete-like features (-1, 1) and add soft noise
        X_bin = torch.sign(torch.randn(num_samples, d))
        noise = torch.randn(num_samples, d) * 0.2
        X = X_bin + noise
    elif 16 <= domain_idx <= 22:
        # Phase 3: NLP & General (Dense clustered distributions)
        # We model clustered topics. Generate 5 cluster centers and assign samples to them
        num_clusters = 5
        centers = torch.randn(num_clusters, d) * 2.0
        cluster_assignments = torch.randint(0, num_clusters, (num_samples,))
        X = centers[cluster_assignments] + torch.randn(num_samples, d) * 0.5
    else:
        # Phase 4: Vision-Flattened (Spatial grid-like features)
        # Model spatial correlations: construct a structured matrix of grid sinusoids
        grid_size = int(np.sqrt(d))
        if grid_size * grid_size != d:
            # If d is not a perfect square, pad/adjust grid representation
            grid_size = 28  # fallback
            d_actual = grid_size * grid_size
        else:
            d_actual = d
        
        X = torch.zeros(num_samples, d_actual)
        freq = (domain_idx - 22) * 0.5  # varying spatial frequencies
        for i in range(num_samples):
            # Phase shifts and combinations of sinusoids
            phase_x = np.random.rand() * 2 * np.pi
            phase_y = np.random.rand() * 2 * np.pi
            grid_x, grid_y = torch.meshgrid(torch.linspace(0, 10, grid_size), torch.linspace(0, 10, grid_size), indexing='ij')
            img = torch.sin(grid_x * freq + phase_x) * torch.cos(grid_y * freq + phase_y)
            X[i] = img.flatten()[:d_actual]
            
        if d_actual != d:
            # Pad or truncate to match d
            if d_actual < d:
                X = torch.cat([X, torch.zeros(num_samples, d - d_actual)], dim=1)
            else:
                X = X[:, :d]

    # Generate a unique task mapping for this domain to simulate distinct target manifolds
    # Use a deterministic orthogonal projection to define the labels
    V = torch.randn(d, out_dim)
    # Perform QR decomposition to get orthogonal projection directions
    q, _ = torch.linalg.qr(V)
    # Compute logits and labels
    logits = torch.matmul(X, q)
    y = torch.argmax(logits, dim=-1)

    # Normalize X to keep spectral properties bounded
    X = X / (torch.norm(X, dim=-1, keepdim=True) + 1e-6)

    # Split into train, val, and test matching paper specifications
    # Paper: Exactly 500 samples per domain -> 400 Train, 50 Val, 50 Test
    if num_samples == 500:
        train_X, val_X, test_X = X[:400], X[400:450], X[450:]
        train_y, val_y, test_y = y[:400], y[400:450], y[450:]
    else:
        # Fallback ratio: 80% / 10% / 10%
        split_tr = int(0.8 * num_samples)
        split_va = int(0.9 * num_samples)
        train_X, val_X, test_X = X[:split_tr], X[split_tr:split_va], X[split_va:]
        train_y, val_y, test_y = y[:split_tr], y[split_tr:split_va], y[split_va:]

    return train_X, train_y, val_X, val_y, test_X, test_y

def get_30_domains(num_samples=500, d=768, out_dim=10, seed=42):
    """
    Returns a dictionary of all 30 domains with train/val/test splits.
    """
    domains = {}
    for i in range(1, 31):
        tr_x, tr_y, va_x, va_y, te_x, te_y = generate_domain_data(i, num_samples, d, out_dim, seed)
        domains[i] = {
            'train_x': tr_x,
            'train_y': tr_y,
            'val_x': va_x,
            'val_y': va_y,
            'test_x': te_x,
            'test_y': te_y
        }
    return domains

if __name__ == '__main__':
    # Simple test of data generation
    print("Testing 30 domains dataset generation...")
    domains = get_30_domains(num_samples=100, d=768, out_dim=10)
    print(f"Generated {len(domains)} domains.")
    for i in [1, 10, 18, 25]:
        print(f"Domain {i}: train_x shape = {domains[i]['train_x'].shape}, train_y shape = {domains[i]['train_y'].shape}")
    print("Successfully generated all domains!")
