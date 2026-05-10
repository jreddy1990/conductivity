"""Generate training curve chart from current + previous run output."""
import re
import sys
import matplotlib.pyplot as plt


def parse_log(log_path):
    steps, train_mae, val_mae = [], [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Step\s+(\d+).*train MAE=(\d+\.\d+).*val MAE=(\d+\.\d+)', line)
            if m:
                steps.append(int(m.group(1)))
                train_mae.append(float(m.group(2)))
                val_mae.append(float(m.group(3)))
    return steps, train_mae, val_mae


def plot(log_path, prev_log_path, out_path):
    steps, train_mae, val_mae = parse_log(log_path)
    prev_steps, prev_train, prev_val = parse_log(prev_log_path)

    if not steps:
        print("No training steps found yet.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, train_mae, 'b-o', markersize=3, label='Train MAE (regularized)', linewidth=1.5)
    ax1.plot(steps, val_mae, 'r-s', markersize=3, label='Val MAE (regularized)', linewidth=1.5)
    if prev_steps:
        ax1.plot(prev_steps, prev_train, 'b--', alpha=0.3, linewidth=1, label='Train (prev, no reg)')
        ax1.plot(prev_steps, prev_val, 'r--', alpha=0.3, linewidth=1, label='Val (prev, no reg)')
    ax1.axhline(y=0.591, color='gray', linestyle='--', linewidth=1, label='MLP baseline (0.591)')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('MAE (mS/cm)', fontsize=12)
    ax1.set_title('MolSet Transformer — Regularized Training', fontsize=13)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylim(0.2, max(1.2, max(val_mae[:3]) * 1.1) if len(val_mae) > 3 else 6.0)
    ax1.set_xlim(0, 5200)
    ax1.grid(True, alpha=0.3)

    ratios = [t / v for t, v in zip(train_mae, val_mae)]
    ax2.plot(steps, ratios, 'g-o', markersize=3, linewidth=1.5, label='Train/Val ratio')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='No overfit (ratio=1.0)')
    ax2.axhline(y=0.63, color='red', linestyle=':', linewidth=1, label='Prev run (0.63)')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Train MAE / Val MAE', fontsize=12)
    ax2.set_title('Overfitting Monitor', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.4, 1.1)
    ax2.set_xlim(0, 5200)
    ax2.grid(True, alpha=0.3)

    best_val = min(val_mae)
    best_step = steps[val_mae.index(best_val)]
    best_ratio = train_mae[val_mae.index(best_val)] / best_val
    fig.suptitle(f'Best val: {best_val:.3f} mS/cm @ step {best_step} | T/V ratio: {best_ratio:.2f}',
                 fontsize=11, y=0.02)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved: {out_path} ({len(steps)} steps, best val={best_val:.3f})")


if __name__ == "__main__":
    plot(sys.argv[1], sys.argv[2], sys.argv[3])
