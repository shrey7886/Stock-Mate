import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_DIR  = Path(__file__).resolve().parents[2]
LOG_DIR   = ROOT_DIR / "lightning_logs" / "tft"
MODEL_DIR = ROOT_DIR / "ml_service" / "models" / "tft_trained"

def plot_loss_curves():
    # Find latest version
    versions = sorted(LOG_DIR.glob("version_*"))
    if not versions:
        print("[ERROR] No tensorboard logs found")
        return

    log_path = str(versions[-1])
    print(f"Reading logs from: {log_path}")

    ea = EventAccumulator(log_path)
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    print(f"Available metrics: {available}")

    train_loss = [(s.step, s.value) for s in ea.Scalars("train_loss_epoch")]
    val_loss   = [(s.step, s.value) for s in ea.Scalars("val_loss")]

    steps_t, values_t = zip(*train_loss)
    steps_v, values_v = zip(*val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(steps_t, values_t, label="Train Loss", marker="o")
    plt.plot(steps_v, values_v, label="Val Loss",   marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("QuantileLoss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = MODEL_DIR / "loss_curve.png"
    plt.savefig(str(out))
    plt.show()
    print(f"Saved → {out}")

    # Print summary
    print(f"\nBest train loss : {min(values_t):.6f} at epoch {steps_t[values_t.index(min(values_t))]}")
    print(f"Best val loss   : {min(values_v):.6f} at epoch {steps_v[values_v.index(min(values_v))]}")
    print(f"Final train loss: {values_t[-1]:.6f}")
    print(f"Final val loss  : {values_v[-1]:.6f}")
    gap = values_t[-1] - values_v[-1]
    print(f"Final gap (train-val): {gap:.6f}")
    if gap > 0.002:
        print("  ✗ Large gap — overfitting present")
    else:
        print("  ✓ Small gap — healthy generalization")

if __name__ == "__main__":
    plot_loss_curves()