import torch
import torch.nn as nn

def picard_solver(f_func, x, max_iter=50, tol=1e-4):
    """
    Solves fixed-point equation z* = f_func(z*, x) using standard Picard iteration.
    Returns:
        z_star: [batch_size, d]
        iterations: Number of iterations taken to converge
    """
    # Initialize z_0 as zeros
    z = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
    
    for i in range(max_iter):
        z_next = f_func(z, x)
        residual = torch.norm(z_next - z, dim=-1).max()
        z = z_next
        if residual < tol:
            return z, i + 1
            
    return z, max_iter

def anderson_solver(f_func, x, max_iter=50, tol=1e-4, m=5):
    """
    Solves fixed-point equation z* = f_func(z*, x) using Anderson Acceleration.
    Returns:
        z_star: [batch_size, d]
        iterations: Number of iterations taken to converge
    """
    bsz, d = x.shape
    z0 = torch.zeros_like(x)
    
    if max_iter <= 2:
        return f_func(z0, x), max_iter

    # Initialize history buffers
    # X: [m, bsz, d]
    # F: [m, bsz, d]
    # G: [m, bsz, d] where G = F - X
    X = torch.zeros(m, bsz, d, device=x.device, dtype=x.dtype)
    F = torch.zeros(m, bsz, d, device=x.device, dtype=x.dtype)
    
    X[0] = z0
    F[0] = f_func(z0, x)
    
    # Second iteration
    X[1] = F[0]
    F[1] = f_func(F[0], x)
    
    G = F - X  # [m, bsz, d]
    
    for k in range(2, max_iter):
        n_prev = min(k, m)
        # Solve least squares for each sample in the batch
        # We want to minimize || \sum_{j=0}^{n_prev-1} \alpha_j G[k-1-j] ||^2 s.t. \sum \alpha_j = 1
        # Let's set up the system. A standard way is to solve:
        # \Delta G^T \Delta G \gamma = \Delta G^T G[-1]
        # where \Delta G_j = G[j+1] - G[j]
        # Then \alpha is reconstructed from \gamma
        
        # Extract the window
        G_window = G[:n_prev]  # [n_prev, bsz, d]
        X_window = X[:n_prev]  # [n_prev, bsz, d]
        F_window = F[:n_prev]  # [n_prev, bsz, d]
        
        # We can construct the least-squares problem batch-wise.
        # Since bsz and n_prev are small, we can solve it efficiently.
        # For simplicity and high numerical stability across batches, we solve:
        # minimize || \sum \alpha_j G_j ||^2 s.t. \sum \alpha = 1
        # Let's rewrite as: G_window is shape [n_prev, bsz, d]. 
        # Permute to [bsz, d, n_prev] to solve for each batch element.
        mat_G = G_window.permute(1, 2, 0)  # [bsz, d, n_prev]
        
        # We want to find vector \alpha of shape [bsz, n_prev, 1]
        # We solve the constrained system:
        # [ mat_G^T mat_G   ones ] [ \alpha   ]   [ 0 ]
        # [ ones^T          0    ] [ \lambda  ] = [ 1 ]
        # This is a linear system of size (n_prev + 1) for each batch element.
        
        # Let's build LHS
        # mat_G^T mat_G has shape [bsz, n_prev, n_prev]
        GtG = torch.bmm(mat_G.transpose(1, 2), mat_G)  # [bsz, n_prev, n_prev]
        
        # Regularize to prevent singular matrix issues
        GtG = GtG + torch.eye(n_prev, device=x.device).unsqueeze(0) * 1e-6
        
        # Construct constraint system
        ones = torch.ones(bsz, n_prev, 1, device=x.device, dtype=x.dtype)
        
        # Construct full system matrix [bsz, n_prev+1, n_prev+1]
        system_mat = torch.cat([
            torch.cat([GtG, ones], dim=2),
            torch.cat([ones.transpose(1, 2), torch.zeros(bsz, 1, 1, device=x.device, dtype=x.dtype)], dim=2)
        ], dim=1)
        
        # RHS [bsz, n_prev+1, 1]
        rhs = torch.zeros(bsz, n_prev + 1, 1, device=x.device, dtype=x.dtype)
        rhs[:, -1, 0] = 1.0
        
        try:
            # Solve using batch linear solver
            sol = torch.linalg.solve(system_mat, rhs)  # [bsz, n_prev+1, 1]
            alphas = sol[:, :n_prev, 0]  # [bsz, n_prev]
        except torch.linalg.LinAlgError:
            # Fallback to simple average if singular
            alphas = torch.ones(bsz, n_prev, device=x.device, dtype=x.dtype) / n_prev
            
        # Compute the new iterate:
        # z_next = \sum \alpha_j F_j
        # alphas is [bsz, n_prev]
        # F_window is [n_prev, bsz, d] -> permute to [bsz, d, n_prev]
        mat_F = F_window.permute(1, 2, 0)  # [bsz, d, n_prev]
        z_next = torch.bmm(mat_F, alphas.unsqueeze(2)).squeeze(2)  # [bsz, d]
        
        # Calculate residual
        residual = torch.norm(z_next - X_window[-1], dim=-1).max()
        
        if residual < tol:
            return z_next, k + 1
            
        # Update history buffer (roll left)
        if k < m:
            X[k] = z_next
            F[k] = f_func(z_next, x)
            G[k] = F[k] - X[k]
        else:
            X = torch.roll(X, -1, dims=0)
            F = torch.roll(F, -1, dims=0)
            G = torch.roll(G, -1, dims=0)
            X[-1] = z_next
            F[-1] = f_func(z_next, x)
            G[-1] = F[-1] - X[-1]
            
    return F[-1], max_iter


class DEQFunction(torch.autograd.Function):
    """
    A custom autograd function that performs memory-efficient implicit backpropagation.
    Solves for the fixed point z* in the forward pass, and uses the Adjoint Method 
    (Implicit Function Theorem) in the backward pass to calculate exact gradients 
    without saving intermediate solver activations.
    """
    @staticmethod
    def forward(ctx, f_func, x, solver_type, max_iter, tol):
        # Choose solver
        if solver_type == 'anderson':
            z_star, iters = anderson_solver(f_func, x, max_iter=max_iter, tol=tol)
        else:
            z_star, iters = picard_solver(f_func, x, max_iter=max_iter, tol=tol)
            
        # Save tensors for backward pass
        ctx.save_for_backward(z_star, x)
        ctx.f_func = f_func
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.iters = iters
        
        return z_star

    @staticmethod
    def backward(ctx, grad_z):
        z_star, x = ctx.saved_tensors
        f_func = ctx.f_func
        max_iter = ctx.max_iter
        tol = ctx.tol
        
        # Detach z_star to compute partial derivatives
        z_star = z_star.detach().requires_grad_(True)
        x = x.detach().requires_grad_(True)
        
        with torch.enable_grad():
            f_out = f_func(z_star, x)
            
        # We need to solve: w^T = w^T * J_z + grad_z
        # which can be solved as a fixed-point problem: w_{t+1} = w_t * J_z + grad_z
        # where w_t * J_z is the Vector-Jacobian Product (VJP) of f_out with respect to z_star.
        
        def adjoint_fixed_point(w, _):
            # Compute VJP: w_t * \nabla_z f(z*, x)
            # Since f_out depends on z_star, torch.autograd.grad allows VJP computation
            vjp = torch.autograd.grad(f_out, z_star, w, retain_graph=True)[0]
            return vjp + grad_z
            
        # Solve for w_star using Picard solver
        # Start adjoint solver from grad_z
        w = grad_z.clone()
        for _ in range(max_iter):
            w_next = adjoint_fixed_point(w, None)
            res = torch.norm(w_next - w, dim=-1).max()
            w = w_next
            if res < tol:
                break
                
        # Now that we have the converged adjoint w*, we can compute the parameter gradients.
        # The total derivative with respect to parameters \theta is: w*^T * \nabla_\theta f(z*, x)
        # In PyTorch, this is computed by running autograd on f_out with w* as the external gradient.
        params = [p for p in f_func.parameters() if p.requires_grad]
        
        if len(params) > 0:
            grads = torch.autograd.grad(f_out, params, w, retain_graph=True, allow_unused=True)
            for p, g in zip(params, grads):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.clone()
                    else:
                        p.grad += g
                        
        # Compute gradient with respect to input x
        grad_x = torch.autograd.grad(f_out, x, w, retain_graph=False)[0]
        
        # Returns: (f_func, x, solver_type, max_iter, tol)
        # Return None for non-tensor arguments and class objects
        return None, grad_x, None, None, None


class ImplicitDEQLayer(nn.Module):
    """
    Wrapper PyTorch module for the Implicit DEQ layer.
    """
    def __init__(self, f_func, solver_type='anderson', max_iter=50, tol=1e-4):
        super().__init__()
        self.f_func = f_func
        self.solver_type = solver_type
        self.max_iter = max_iter
        self.tol = tol
        self.last_iterations = 0

    def forward(self, x):
        # We call our custom autograd function
        z_star = DEQFunction.apply(self.f_func, x, self.solver_type, self.max_iter, self.tol)
        return z_star
