import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_generator import get_30_domains
from models import System25Model, WideSystem25Model, System3Model
from evaluate import run_lifelong_experiment, welch_t_test


def sequential_bwt_curve(R_matrix):
    """Per-task backward transfer curve computed from the measured R matrix."""
    n = R_matrix.shape[0]
    curve = np.zeros(n)
    for k in range(1, n):
        curve[k] = np.mean([R_matrix[k, i] - R_matrix[i, i] for i in range(k)]) * 100.0
    return curve


def fmt_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return f"{values[0]:+.1f}%"
    return f"{values.mean():+.1f}% ± {values.std(ddof=1):.1f}%"


def fmt_vram(values):
    values = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(values)):
        return "n/a (CPU)"
    return f"{np.nanmean(values):.1f} GB"


def main():
    parser = argparse.ArgumentParser(description="System 3: Sparse Implicit Mixtures lifelong benchmark")
    parser.add_argument('--seeds', type=int, default=5, help='number of random seeds (paper: 5)')
    parser.add_argument('--domains', type=int, default=30, help='number of sequential domains (paper: 30)')
    parser.add_argument('--samples', type=int, default=500, help='samples per domain (paper: 500)')
    parser.add_argument('--epochs', type=int, default=2, help='epochs per domain')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4, help='AdamW learning rate (paper: 1e-4)')
    parser.add_argument('--lambda-ewc', type=float, default=15.0)
    parser.add_argument('--d', type=int, default=768, help='implicit dimension (paper: 768)')
    parser.add_argument('--d-wide', type=int, default=3072, help='wide baseline dimension (paper: 3072)')
    parser.add_argument('--out-dim', type=int, default=10)
    parser.add_argument('--data-seed', type=int, default=42, help='fixed benchmark data seed')
    parser.add_argument('--tau-spawn', type=float, default=0.07,
                        help='R2P novelty threshold (calibrated from measured domain separability)')
    parser.add_argument('--output', type=str, default='evaluation_results.png')
    args = parser.parse_args()

    print("==================================================================")
    print(" SYSTEM 3: SPARSE IMPLICIT MIXTURES LIFE-LONG REASONING PIPELINE ")
    print("==================================================================")
    print(f"Config: {args.domains} domains x {args.samples} samples, {args.seeds} seed(s), "
          f"d={args.d}, epochs={args.epochs}, lr={args.lr}")

    # 1. Benchmark data is fixed across seeds; model init / training vary per seed.
    print("\n[Step 1] Generating sequential reasoning dataset...")
    domains_all = get_30_domains(num_samples=args.samples, d=args.d, out_dim=args.out_dim, seed=args.data_seed)
    domains = {i: domains_all[i] for i in range(1, args.domains + 1)}
    print(f"--> Generated {len(domains)} sequential domains.")

    arch_specs = [
        ('System 2.5 (d=%d)' % args.d, 'sys25'),
        ('Wide Sys 2.5 (d=%d)' % args.d_wide, 'wide'),
        ('System 3 (Ours)', 'sys3'),
    ]
    results = {key: [] for _, key in arch_specs}

    # 2. Run every architecture across every seed
    for seed_idx in range(args.seeds):
        seed = 1000 + seed_idx
        print(f"\n================ SEED {seed_idx + 1}/{args.seeds} (torch seed {seed}) ================")
        for arch_name, key in arch_specs:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if key == 'sys25':
                model = System25Model(d=args.d, out_dim=args.out_dim, solver_type='anderson')
                is_sys3 = False
            elif key == 'wide':
                model = WideSystem25Model(d_in=args.d, d_wide=args.d_wide, out_dim=args.out_dim, solver_type='anderson')
                is_sys3 = False
            else:
                model = System3Model(d=args.d, out_dim=args.out_dim, solver_type='anderson',
                                     tau_spawn=args.tau_spawn)
                is_sys3 = True

            print(f"\n--- {arch_name} (seed {seed}) ---")
            res = run_lifelong_experiment(
                model=model,
                domains=domains,
                is_system3=is_sys3,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lambda_ewc=args.lambda_ewc,
            )
            results[key].append(res)

    # 3. Aggregate measured metrics across seeds
    print("\n" + "=" * 90)
    print("             FINAL COMPARATIVE EVALUATION RESULTS (measured, mean ± std over seeds)")
    print("=" * 90)
    print(f"| {'Architecture':<22} | {'Final BWT':<16} | {'Final FWT':<16} | {'Peak VRAM':<12} | {'Experts':<8} |")
    print(f"|{'-'*24}|{'-'*18}|{'-'*18}|{'-'*14}|{'-'*10}|")

    agg = {}
    for arch_name, key in arch_specs:
        runs = results[key]
        bwts = [r['final_bwt'] for r in runs]
        fwts = [r['final_fwt'] for r in runs]
        vrams = [r['vram_history'][-1] for r in runs]
        experts = [r['expert_counts'][-1] for r in runs]
        agg[key] = {'bwt': bwts, 'fwt': fwts}
        print(f"| {arch_name:<22} | {fmt_mean_std(bwts):<16} | {fmt_mean_std(fwts):<16} | "
              f"{fmt_vram(vrams):<12} | {int(np.mean(experts)):<8} |")
    print("=" * 90)

    # Welch's t-tests (System 3 vs baselines), meaningful when seeds >= 2
    if args.seeds >= 2:
        print("\nWelch's t-tests (System 3 vs baselines):")
        for arch_name, key in arch_specs[:-1]:
            t_b, p_b = welch_t_test(agg['sys3']['bwt'], agg[key]['bwt'])
            t_f, p_f = welch_t_test(agg['sys3']['fwt'], agg[key]['fwt'])
            print(f"  vs {arch_name}: BWT t={t_b:.2f}, p={p_b:.3f} | FWT t={t_f:.2f}, p={p_f:.3f}")
    else:
        print("\n[Note] Single-seed run: std and Welch's t-tests require --seeds >= 2 (paper uses 5).")

    # 4. Visualization from measured data (first seed's trajectories)
    print("\n[Step 4] Generating comparative performance visualizations...")
    res_25 = results['sys25'][0]
    res_3 = results['sys3'][0]
    n = args.domains
    domains_range = np.arange(1, n + 1)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Measured average solver iterations
    axs[0, 0].plot(domains_range, res_25['convergence_history'], color='#808080', linewidth=2.5, label='System 2.5 (Dense DEQ)')
    axs[0, 0].plot(domains_range, res_3['convergence_history'], color='#008080', linewidth=3.0, label='System 3 (Ours)')
    axs[0, 0].set_title('(a) Average Anderson Solver Iterations (measured)', fontsize=13, fontweight='bold', pad=10)
    axs[0, 0].set_xlabel(f'Domains (1 to {n})', fontsize=11)
    axs[0, 0].set_ylabel('Solver Iterations (avg.)', fontsize=11)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].legend(fontsize=10)

    # (b) Sequential BWT computed from the measured R matrices
    axs[0, 1].plot(domains_range, sequential_bwt_curve(res_25['R_matrix']), color='#808080', linewidth=2.5, label='System 2.5 (Dense DEQ)')
    axs[0, 1].plot(domains_range, sequential_bwt_curve(res_3['R_matrix']), color='#008080', linewidth=3.0, label='System 3 (Ours)')
    axs[0, 1].set_title('(b) Sequential Backward Transfer (measured)', fontsize=13, fontweight='bold', pad=10)
    axs[0, 1].set_xlabel(f'Domains (1 to {n})', fontsize=11)
    axs[0, 1].set_ylabel('Backward Transfer (BWT %)', fontsize=11)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].legend(fontsize=10)

    # (c) Measured peak VRAM per task (NaN when running on CPU)
    axs[1, 0].plot(domains_range, res_25['vram_history'], color='#808080', linestyle='--', linewidth=2.0, label='System 2.5 (Dense DEQ)')
    axs[1, 0].plot(domains_range, res_3['vram_history'], color='#008080', linewidth=3.0, label='System 3 (Ours)')
    axs[1, 0].set_title('(c) Peak VRAM per Task (measured)', fontsize=13, fontweight='bold', pad=10)
    axs[1, 0].set_xlabel(f'Domains (1 to {n})', fontsize=11)
    axs[1, 0].set_ylabel('Peak VRAM Allocation (GB)', fontsize=11)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend(fontsize=10)

    # (d) Actual expert recruitment profile
    axs[1, 1].step(domains_range, res_3['expert_counts'], color='#008080', where='mid', linewidth=3.0, label='System 3 Spawned Experts')
    axs[1, 1].plot(domains_range, [1] * n, color='#808080', linestyle='--', linewidth=2.0, label='System 2.5 (Dense DEQ)')
    axs[1, 1].set_title('(d) Expert Dynamic Spawning (R2P, measured)', fontsize=13, fontweight='bold', pad=10)
    axs[1, 1].set_xlabel(f'Domains (1 to {n})', fontsize=11)
    axs[1, 1].set_ylabel('Active Expert Count (M)', fontsize=11)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].legend(fontsize=10)

    plt.suptitle("System 3 Lifelong Reasoning: Measured Results", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"--> Visualization saved to: {args.output}")
    print("\nLifelong sequential reasoning experiment completed.")


if __name__ == '__main__':
    main()
