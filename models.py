import torch
import torch.nn as nn
from deq_solver import ImplicitDEQLayer
from router import ContrastiveRouter, scale_to_contractive

class SingleExpertTransition(nn.Module):
    """
    Expert transition operator: f(z, x) = tanh(W_z * z + W_x * x + b)
    Enforced to be strictly contractive at initialization.
    """
    def __init__(self, d=768, margin=0.95):
        super().__init__()
        self.d = d
        self.W_z = nn.Parameter(torch.randn(d, d) * (1.0 / d**0.5))
        self.W_x = nn.Parameter(torch.randn(d, d) * (1.0 / d**0.5))
        self.b = nn.Parameter(torch.zeros(d))
        
        # Enforce contractivity right away
        scale_to_contractive(self.W_z, margin=margin)

    def forward(self, z, x):
        # z: [batch_size, d]
        # x: [batch_size, d]
        return torch.tanh(F_linear(z, self.W_z) + F_linear(x, self.W_x) + self.b)

def F_linear(input_tensor, weight):
    """ Helper to perform matrix multiplication with Parameter weight """
    return torch.matmul(input_tensor, weight.t())


class System25Model(nn.Module):
    """
    System 2.5: Dense single-weight implicit reasoner (DEQ).
    Protected by standard FP-EWC.
    """
    def __init__(self, d=768, out_dim=10, solver_type='anderson', max_iter=50, tol=1e-4, margin=0.95):
        super().__init__()
        self.d = d
        self.margin = margin
        self.transition = SingleExpertTransition(d, margin=margin)
        self.deq_layer = ImplicitDEQLayer(self.transition, solver_type=solver_type, max_iter=max_iter, tol=tol)
        
        # Shared task head
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        # x: [batch_size, d]
        z_star = self.deq_layer(x)  # Solve fixed point
        logits = self.head(z_star)
        return logits, z_star


class WideTransition(nn.Module):
    """
    Transition operator in wide space: f(z, x_proj) = tanh(W_z * z + x_proj + b)
    """
    def __init__(self, d_wide=3072, margin=0.95):
        super().__init__()
        self.d_wide = d_wide
        self.W_z = nn.Parameter(torch.randn(d_wide, d_wide) * (1.0 / d_wide**0.5))
        self.b = nn.Parameter(torch.zeros(d_wide))
        scale_to_contractive(self.W_z, margin=margin)

    def forward(self, z, x_proj):
        return torch.tanh(torch.matmul(z, self.W_z.t()) + x_proj + self.b)


class WideSystem25Model(nn.Module):
    """
    Wide System 2.5: Single-weight DEQ scaled to d = 3072 to match parameter capacity.
    """
    def __init__(self, d_in=768, d_wide=3072, out_dim=10, solver_type='anderson', max_iter=50, tol=1e-4, margin=0.95):
        super().__init__()
        self.d_in = d_in
        self.d_wide = d_wide
        self.margin = margin
        
        # Input projection layer to lift features to wide dimension
        self.proj_in = nn.Linear(d_in, d_wide)
        
        # Transition operator in wide space
        self.transition = WideTransition(d_wide, margin=margin)
        self.deq_layer = ImplicitDEQLayer(self.transition, solver_type=solver_type, max_iter=max_iter, tol=tol)
        
        # Wide task head
        self.head = nn.Linear(d_wide, out_dim)

    def forward(self, x):
        # Project input first
        x_proj = self.proj_in(x)
        # Solve fixed point
        z_star = self.deq_layer(x_proj)
        logits = self.head(z_star)
        return logits, z_star


class System3Model(nn.Module):
    """
    System 3: Sparse Implicit Mixture-of-Experts (MoE) DEQ.
    Uses Contractive Gated Mixture (CGM) to guarantee stability.
    """
    def __init__(self, d=768, out_dim=10, solver_type='anderson', max_iter=50, tol=1e-4, margin=0.95):
        super().__init__()
        self.d = d
        self.solver_type = solver_type
        self.max_iter = max_iter
        self.tol = tol
        self.margin = margin
        
        # Contrastive Router
        self.router = ContrastiveRouter(d_in=d, d_r=128, tau_spawn=0.8, top_k=2)
        
        # Expert pool (stored in a ModuleList)
        self.experts = nn.ModuleList()
        
        # Shared task head
        self.head = nn.Linear(d, out_dim)

        # Persistent DEQ solver wrapper (transition supplied per forward call)
        self.deq_layer = ImplicitDEQLayer(None, solver_type=solver_type, max_iter=max_iter, tol=tol)

        # Track dynamic spawns for optimizer updates
        self.new_expert_spawned = False

        # Spawn the first expert initially
        self.spawn_new_expert()

    def spawn_new_expert(self, prototype_embed=None):
        """
        Dynamically spawns a new expert and registers its prototype in the router.
        """
        # Create a new expert with C-FIRE contractivity
        new_expert = SingleExpertTransition(d=self.d, margin=self.margin)
        
        # Explicitly move to the correct device
        if len(self.experts) > 0:
            device = next(self.experts[0].parameters()).device
            new_expert.to(device)
        elif next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            new_expert.to(device)
            
        self.experts.append(new_expert)
        
        # Register in router
        if prototype_embed is None:
            # First expert initialization dummy prototype
            device = next(self.parameters()).device if next(self.parameters(), None) is not None else torch.device('cpu')
            prototype_embed = torch.randn(128, device=device)
        self.router.add_prototype(prototype_embed)
        self.new_expert_spawned = True
        
        print(f"--> [System 3] Expert {len(self.experts)} Spawned! Total experts = {len(self.experts)}")

    def get_cgm_transition(self, gates):
        r"""
        Returns a function representing the Contractive Gated Mixture (CGM) operator.
        z_next = \sum_{i} g_i(x) f_{\theta_i}(z, x)
        This is z-independent routing which guarantees strict contraction properties.
        """
        def cgm_transition_fn(z, x):
            # z: [batch_size, d]
            # x: [batch_size, d]
            z_next = torch.zeros_like(z)
            
            # Efficiently compute only for active experts per sample
            for i, expert in enumerate(self.experts):
                expert_gates = gates[:, i]
                # Identify which batch elements route to expert i
                active_idx = torch.where(expert_gates > 0.0)[0]
                if len(active_idx) > 0:
                    z_expert = expert(z[active_idx], x[active_idx])
                    # Accumulate gated expert state
                    z_next[active_idx] += expert_gates[active_idx].unsqueeze(1) * z_expert
                    
            return z_next
            
        return cgm_transition_fn

    def forward(self, x, training=True):
        # 1. Run the Router to obtain gates and novelty info
        gates, spawn_expert, mean_embed, routing_loss = self.router(x, training=training)
        
        # 2. If novelty threshold triggers spawning, perform recruitment
        if training and spawn_expert:
            self.spawn_new_expert(mean_embed)
            # Recompute similarities and gates with the newly added expert
            gates, _, _, routing_loss = self.router(x, training=training)
            
        # 3. Instantiate the CGM transition operator using the computed gates.
        # Gates depend only on x (z-independent), so contractivity (Prop. 1) holds;
        # keeping them attached lets the router learn from the task loss.
        transition_fn = self.get_cgm_transition(gates)

        # 4. Solve for the fixed point z* through the persistent DEQ layer
        z_star = self.deq_layer(x, f_func=transition_fn)
        
        # 5. Compute task logits
        logits = self.head(z_star)
        
        return logits, z_star, routing_loss, gates
