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

    # History buffers: X (iterates), F (function values), G = F - X (residuals)
    X = torch.zeros(m, bsz, d, device=x.device, dtype=x.dtype)
    F = torch.zeros(m, bsz, d, device=x.device, dtype=x.dtype)

    X[0] = z0
    F[0] = f_func(z0, x)

    X[1] = F[0]
    F[1] = f_func(F[0], x)

    G = F - X  # [m, bsz, d]

    for k in range(2, max_iter):
        n_prev = min(k, m)
        # Solve the constrained least-squares problem per batch element:
        # minimize || sum_j alpha_j G_j ||^2  s.t.  sum_j alpha_j = 1
        G_window = G[:n_prev]  # [n_prev, bsz, d]
        X_window = X[:n_prev]
        F_window = F[:n_prev]

        mat_G = G_window.permute(1, 2, 0)  # [bsz, d, n_prev]

        # KKT system:
        # [ G^T G   1 ] [ alpha  ]   [ 0 ]
        # [ 1^T     0 ] [ lambda ] = [ 1 ]
        GtG = torch.bmm(mat_G.transpose(1, 2), mat_G)  # [bsz, n_prev, n_prev]
        GtG = GtG + torch.eye(n_prev, device=x.device).unsqueeze(0) * 1e-6

        ones = torch.ones(bsz, n_prev, 1, device=x.device, dtype=x.dtype)
        system_mat = torch.cat([
            torch.cat([GtG, ones], dim=2),
            torch.cat([ones.transpose(1, 2), torch.zeros(bsz, 1, 1, device=x.device, dtype=x.dtype)], dim=2)
        ], dim=1)

        rhs = torch.zeros(bsz, n_prev + 1, 1, device=x.device, dtype=x.dtype)
        rhs[:, -1, 0] = 1.0

        try:
            sol = torch.linalg.solve(system_mat, rhs)  # [bsz, n_prev+1, 1]
            alphas = sol[:, :n_prev, 0]  # [bsz, n_prev]
        except torch.linalg.LinAlgError:
            alphas = torch.ones(bsz, n_prev, device=x.device, dtype=x.dtype) / n_prev

        # z_next = sum_j alpha_j F_j
        mat_F = F_window.permute(1, 2, 0)  # [bsz, d, n_prev]
        z_next = torch.bmm(mat_F, alphas.unsqueeze(2)).squeeze(2)  # [bsz, d]

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


class ImplicitDEQLayer(nn.Module):
    """
    Implicit DEQ layer with O(1)-memory IFT (adjoint) backpropagation.

    Forward: solves z* = f(z*, x) under torch.no_grad(), then re-applies f once
    with gradients enabled so autograd records the dependency of z* on the
    parameters of f and on x.

    Backward: a hook on z* replaces the incoming gradient dL/dz* with the
    adjoint w* solving  w = dL/dz* + J_z^T w  (Neumann/Picard iteration), so the
    subsequent autograd step through the single recorded application of f yields
    the exact implicit gradients  w*^T (df/dtheta)  and  w*^T (df/dx)  without
    unrolling the solver.
    """
    def __init__(self, f_func, solver_type='anderson', max_iter=50, tol=1e-4):
        super().__init__()
        self.f_func = f_func
        self.solver_type = solver_type
        self.max_iter = max_iter
        self.tol = tol
        self.last_iterations = 0

    def forward(self, x, f_func=None):
        f = f_func if f_func is not None else self.f_func
        solver = anderson_solver if self.solver_type == 'anderson' else picard_solver

        # 1. Solve for the fixed point without tracking gradients
        with torch.no_grad():
            z_star, iters = solver(f, x, max_iter=self.max_iter, tol=self.tol)
        self.last_iterations = iters

        if not torch.is_grad_enabled():
            return z_star

        # 2. Re-engage autograd with a single application of f at the fixed point
        z_star = z_star.detach()
        z_attached = f(z_star, x)

        # 3. Prepare the adjoint solve for the backward pass
        z0 = z_star.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            f0 = f(z0, x)

        max_iter, tol = self.max_iter, self.tol

        def adjoint_hook(grad):
            # Solve w = grad + J_z^T w via Picard iteration (guaranteed to
            # converge because f is contractive in z).
            w = grad.clone()
            for _ in range(max_iter):
                w_next = torch.autograd.grad(f0, z0, w, retain_graph=True)[0] + grad
                res = torch.norm(w_next - w, dim=-1).max()
                w = w_next
                if res < tol:
                    break
            return w

        if z_attached.requires_grad:
            z_attached.register_hook(adjoint_hook)
        return z_attached
