import torch
import torch.nn as nn
import torch.nn.functional as F

def scale_to_contractive(weight, margin=0.95, n_power_iterations=10):
    """
    C-FIRE spectral normalization step (paper Algorithm 2).
    Estimates the spectral norm sigma_max of the weight matrix via power
    iteration and rescales W <- W / max(1, sigma_max / margin), enforcing
    L <= margin < 1 (Banach contractivity).
    """
    with torch.no_grad():
        u = torch.randn(weight.shape[0], 1, device=weight.device, dtype=weight.dtype)
        u = u / (torch.norm(u) + 1e-12)
        v = None
        for _ in range(n_power_iterations):
            v = torch.matmul(weight.t(), u)
            v = v / (torch.norm(v) + 1e-12)
            u = torch.matmul(weight, v)
            u_norm = torch.norm(u)
            u = u / (u_norm + 1e-12)
        sigma = torch.matmul(u.t(), torch.matmul(weight, v)).item()
        if sigma > margin:
            weight.mul_(margin / sigma)
    return weight

class ContrastiveRouter(nn.Module):
    """
    Contrastive Router with Router Recruitment Policy (R2P) and Load Balancing Loss.
    Uses input similarity to dynamically route queries and recruit new experts.
    """
    def __init__(self, d_in=768, d_r=128, tau_spawn=0.8, top_k=2, temp=1.0, freeze_features=True):
        super().__init__()
        self.d_in = d_in
        self.d_r = d_r
        self.tau_spawn = tau_spawn
        self.top_k = top_k
        self.temp = temp
        # When True, the projection and prototypes are frozen so routing is a
        # fixed deterministic function of x for the whole stream. A trainable
        # router drifts during continual training, which (measured, Run 002/003)
        # re-triggers novelty spawning and re-routes old domains away from the
        # experts that learned them, erasing the benefit of expert isolation.
        # Freezing extends the paper's post-hoc router-freeze rationale
        # (Theorem 1) to the full stream.
        self.freeze_features = freeze_features

        # Router projection layer (R(x))
        self.projection = nn.Sequential(
            nn.Linear(d_in, d_r),
            nn.ReLU(),
            nn.Linear(d_r, d_r)
        )
        if freeze_features:
            for p in self.projection.parameters():
                p.requires_grad_(False)
        # List of expert prototype vectors (c_i)
        self.prototypes = nn.ParameterList()

    def add_prototype(self, embed):
        """
        Dynamically adds a new prototype vector c_{M+1} to the router.
        """
        # Normalize to unit length
        proto_val = embed.detach().clone()
        proto_val = proto_val / (torch.norm(proto_val) + 1e-6)
        new_proto = nn.Parameter(proto_val, requires_grad=not self.freeze_features)
        self.prototypes.append(new_proto)
        
    def get_num_experts(self):
        return len(self.prototypes)

    def compute_similarities(self, x):
        """
        Computes cosine similarities between query embeddings and all expert prototypes.
        Returns:
            similarities: [batch_size, M]
            r: [batch_size, d_r] (query embeddings)
        """
        r = self.projection(x)  # [batch_size, d_r]
        r_norm = r / (torch.norm(r, dim=-1, keepdim=True) + 1e-6)
        
        M = len(self.prototypes)
        if M == 0:
            return torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype), r_norm
            
        # Stack prototypes and normalize
        protos = torch.stack(list(self.prototypes))  # [M, d_r]
        protos_norm = protos / (torch.norm(protos, dim=-1, keepdim=True) + 1e-6)
        
        # Cosine similarity: [batch_size, M]
        sims = torch.matmul(r_norm, protos_norm.t())
        return sims, r_norm

    def forward(self, x, training=True):
        """
        Performs novelty detection, expert recruitment, and routing.
        Returns:
            gates: [batch_size, M] gating weights (sparse, sum to 1)
            spawn_expert: bool, whether a new expert should be spawned
            mean_embed: [d_r] embedding vector for new prototype if spawned
            routing_loss: scalar tensor of auxiliary balancing & z-loss
        """
        sims, r_norm = self.compute_similarities(x)
        M = sims.shape[1]
        bsz = x.shape[0]
        
        # Check novelty / recruitment
        spawn_expert = False
        mean_embed = None
        
        if M == 0:
            # Must spawn the first expert
            spawn_expert = True
            mean_embed = r_norm.mean(dim=0)
            # Dummy values for routing
            gates = torch.ones(bsz, 1, device=x.device, dtype=x.dtype)
            routing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return gates, spawn_expert, mean_embed, routing_loss

        # Compute max similarity per sample
        max_sims, _ = sims.max(dim=-1)
        mean_max_sim = max_sims.mean().item()
        
        # If novelty is high (average maximum similarity < threshold)
        if training and mean_max_sim < self.tau_spawn:
            spawn_expert = True
            mean_embed = r_norm.mean(dim=0)
            
        # Perform top-k routing
        # If M < top_k, we clip k to M
        k = min(self.top_k, M)
        
        # Get top-k indices and values
        top_k_sims, top_k_indices = torch.topk(sims, k, dim=-1)
        
        # Gating Softmax over top-k (scaled by temperature)
        g_scores = F.softmax(top_k_sims / self.temp, dim=-1)  # [batch_size, k]
        
        # Scatter back to full M-dim gates
        gates = torch.zeros_like(sims)
        gates.scatter_(1, top_k_indices, g_scores)
        
        # Calculate Load Balancing and z-loss (only if training)
        if training and M > 1:
            # 1. Load balancing loss: f_i is fraction of routing to expert i
            # Construct top-k binary mask
            mask = torch.zeros_like(sims)
            mask.scatter_(1, top_k_indices, 1.0)
            f = mask.mean(dim=0)  # [M]
            P = gates.mean(dim=0)  # [M]
            
            lbl_loss = M * torch.sum(f * P)
            
            # 2. Router z-loss (penalize large logits/similarities before softmax)
            # Formula: beta * mean( (log sum exp (s_i)) ^ 2 )
            # Since sims is scaled cosine similarity, let's compute:
            z_loss = torch.mean(torch.logsumexp(sims, dim=-1) ** 2)
            
            alpha = 0.01
            beta = 0.001
            routing_loss = alpha * lbl_loss + beta * z_loss
        else:
            routing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
        return gates, spawn_expert, mean_embed, routing_loss
