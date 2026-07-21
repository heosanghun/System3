import torch
import numpy as np
from trainer import train_single_domain, EWCManager

def evaluate_model_on_task(model, test_x, test_y, is_system3=False, batch_size=256):
    """
    Evaluates a model's accuracy on a specific task's test set.
    Returns:
        acc: accuracy on the test set
        avg_iters: average solver iterations taken during the forward passes
    """
    model.eval()
    device = next(model.parameters()).device

    test_x = test_x.to(device)
    test_y = test_y.to(device)

    correct = 0
    total = test_y.shape[0]
    iter_records = []

    with torch.no_grad():
        for start in range(0, total, batch_size):
            bx = test_x[start:start + batch_size]
            by = test_y[start:start + batch_size]
            if is_system3:
                logits, z_star, _, _ = model(bx, training=False)
            else:
                logits, z_star = model(bx)
            iter_records.append(model.deq_layer.last_iterations)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()

    acc = correct / total
    avg_iters = float(np.mean(iter_records)) if iter_records else float('nan')
    return acc, avg_iters

def run_lifelong_experiment(model, domains, is_system3=False, epochs=3, batch_size=32, lr=1e-4, lambda_ewc=15.0):
    """
    Runs the complete N-domain sequential lifelong training experiment.
    All reported quantities are measured, never simulated:
      - R_matrix[k, i]: test accuracy on task i after training on task k.
      - expert_counts: actual number of experts in the model after each task.
      - vram_history: torch.cuda.max_memory_allocated() per task (NaN on CPU).
      - convergence_history: average measured solver iterations per task.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_tasks = len(domains)
    R_matrix = np.zeros((num_tasks, num_tasks))
    expert_counts = []
    vram_history = []
    convergence_history = []

    # Initialize EWC Manager
    ewc_manager = EWCManager(model, lambda_ewc=lambda_ewc)

    # Independent zero-shot baseline accuracy per task (untrained model),
    # used by the FWT definition of Lopez-Paz & Ranzato (2017).
    baselines = np.zeros(num_tasks)
    for i in range(num_tasks):
        baselines[i], _ = evaluate_model_on_task(model, domains[i+1]['test_x'], domains[i+1]['test_y'], is_system3=is_system3)

    print(f"\nStarting lifelong training for {'System 3 (MoE DEQ)' if is_system3 else 'System 2.5 (Dense DEQ)'}...")

    for k in range(num_tasks):
        task_idx = k + 1
        train_x = domains[task_idx]['train_x']
        train_y = domains[task_idx]['train_y']

        # Reset peak VRAM before task training
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Train on the current domain
        loss = train_single_domain(
            model=model,
            train_x=train_x,
            train_y=train_y,
            test_x=domains[task_idx]['test_x'],
            test_y=domains[task_idx]['test_y'],
            domain_idx=task_idx,
            ewc_manager=ewc_manager,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            is_system3=is_system3
        )

        # Actual expert count in the model (1 for dense architectures)
        num_exp = len(model.experts) if is_system3 else 1
        expert_counts.append(num_exp)

        # Measured peak VRAM for this task (GB); NaN when running on CPU
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
        else:
            peak_vram = float('nan')
        vram_history.append(peak_vram)

        # Evaluate on all tasks trained so far.
        # R_matrix[k, i] is accuracy on task i after training on task k.
        task_iter_records = []
        for i in range(k + 1):
            acc, iters = evaluate_model_on_task(model, domains[i+1]['test_x'], domains[i+1]['test_y'], is_system3=is_system3)
            R_matrix[k, i] = acc
            task_iter_records.append(iters)

        # FWT step: zero-shot accuracy on the next unseen task k+1
        if k < num_tasks - 1:
            fwt_acc, _ = evaluate_model_on_task(model, domains[k+2]['test_x'], domains[k+2]['test_y'], is_system3=is_system3)
            R_matrix[k, k+1] = fwt_acc

        # Measured average solver iterations across this task's evaluations
        iters = float(np.mean(task_iter_records))
        convergence_history.append(iters)

        vram_str = f"{peak_vram:.1f} GB" if not np.isnan(peak_vram) else "n/a (CPU)"
        print(f"Task {task_idx:02d}/{num_tasks} | Loss: {loss:.4f} | Active Experts: {num_exp} | Peak VRAM: {vram_str} | Avg Iterations: {iters:.1f}")

    # Final metrics at task N
    # 1. Backward Transfer (BWT): mean_i (R[N-1, i] - R[i, i])
    bwt_list = []
    for i in range(num_tasks - 1):
        bwt_list.append(R_matrix[num_tasks-1, i] - R_matrix[i, i])
    final_bwt = np.mean(bwt_list) * 100.0  # as percentage

    # 2. Forward Transfer (FWT): mean_i (R[i-1, i] - baselines[i])
    fwt_list = []
    for i in range(1, num_tasks):
        fwt_list.append(R_matrix[i-1, i] - baselines[i])
    final_fwt = np.mean(fwt_list) * 100.0  # as percentage

    # 3. Average final accuracy
    final_acc = np.mean(R_matrix[num_tasks-1, :num_tasks]) * 100.0

    return {
        'R_matrix': R_matrix,
        'final_bwt': final_bwt,
        'final_fwt': final_fwt,
        'final_acc': final_acc,
        'expert_counts': expert_counts,
        'vram_history': vram_history,
        'convergence_history': convergence_history,
        'baselines': baselines,
    }

def welch_t_test(sample_a, sample_b):
    """
    Two-sided Welch's t-test (unequal variances) implemented with numpy.
    Returns (t_statistic, p_value). Requires len >= 2 in each sample.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan'), float('nan')
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    if se2 == 0:
        return float('nan'), float('nan')
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    # Welch-Satterthwaite degrees of freedom
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # Two-sided p-value via the regularized incomplete beta function,
    # computed with a continued-fraction expansion (no scipy dependency).
    x = df / (df + t * t)
    p = _reg_inc_beta(df / 2.0, 0.5, x)
    return float(t), float(min(max(p, 0.0), 1.0))

def _reg_inc_beta(a, b, x, max_iter=200, eps=3e-12):
    """Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction."""
    import math
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the symmetry relation where the continued fraction converges fastest
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _reg_inc_beta(b, a, 1.0 - x, max_iter=max_iter, eps=eps)
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - ln_beta) / a
    # Lentz's algorithm
    f, c, d = 1.0, 1.0, 0.0
    for i in range(max_iter * 2):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < eps:
            break
    return front * (f - 1.0)
