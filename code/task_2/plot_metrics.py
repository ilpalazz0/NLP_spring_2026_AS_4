import argparse
import os
import sys

import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Task 2 training metrics")
    parser.add_argument("--metrics_file", type=str, required=True)
    return parser.parse_args()


def read_metrics(metrics_file):
    epochs = []
    train_loss = []
    val_loss = []
    val_em = []
    val_f1 = []
    test_loss = []
    test_em = []
    test_f1 = []

    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                continue
            parts = line.split("\t")
            if len(parts) not in {5, 8}:
                continue
            epochs.append(int(parts[0]))
            train_loss.append(float(parts[1]))
            val_loss.append(float(parts[2]))
            val_em.append(float(parts[3]))
            val_f1.append(float(parts[4]))
            if len(parts) == 8:
                test_loss.append(float(parts[5]))
                test_em.append(float(parts[6]))
                test_f1.append(float(parts[7]))

    return epochs, train_loss, val_loss, val_em, val_f1, test_loss, test_em, test_f1


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.metrics_file)

    epochs, train_loss, val_loss, val_em, val_f1, test_loss, test_em, test_f1 = read_metrics(args.metrics_file)
    if not epochs:
        raise ValueError(f"No epoch records found in {args.metrics_file}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
    if len(test_loss) == len(epochs):
        plt.plot(epochs, test_loss, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_em, marker="o", label="Validation EM")
    plt.plot(epochs, val_f1, marker="o", label="Validation F1")
    if len(test_em) == len(epochs):
        plt.plot(epochs, test_em, marker="o", label="Test EM")
    if len(test_f1) == len(epochs):
        plt.plot(epochs, test_f1, marker="o", label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation EM and F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    score_path = os.path.join(out_dir, "score_curve.png")
    plt.tight_layout()
    plt.savefig(score_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved: {loss_path}")
    print(f"[INFO] Saved: {score_path}")


if __name__ == "__main__":
    main()
