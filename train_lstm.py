from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from recommender_utils import (
    PAD_ID,
    build_item_vocab,
    build_optimizer,
    chronological_split,
    evaluate_loss,
    keep_top_items,
    load_transactions,
    make_loader,
    make_recommendation_frame,
    make_time_window_samples,
    plot_loss_curves,
    plot_metric_bars,
    predict_logits,
    set_seed,
    topk_metrics,
    vectorize_samples,
)


class LSTMRecommender(nn.Module):
    """LSTM-based sequential recommender.

    Uses the final hidden state of an LSTM to predict future item preferences.
    Good at capturing short-to-medium term sequential patterns.
    """

    def __init__(self, num_items: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.head(out[:, -1, :])
        logits[:, PAD_ID] = -1e9
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM recommender with 4 optimizers (SGD/Adam/RMSprop/Adagrad).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    # Data windows (kept consistent across models for fair comparison)
    parser.add_argument("--top-items", type=int, default=500)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--future-hours", type=int, default=24)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--min-history-events", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--top-k", type=int, default=10)

    # LSTM-specific hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--optimizers",
        type=str,
        default="sgd,adam,rmsprop,adagrad",
        help="Comma-separated list of optimizers to evaluate",
    )
    return parser.parse_args()


def train_one_run(args, splits, num_items: int, optimizer_name: str, device):
    x_train, y_train, _, _ = splits["train"]
    x_val, y_val, _, _ = splits["val"]
    x_test, y_test, test_users, test_times = splits["test"]

    model = LSTMRecommender(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = build_optimizer(optimizer_name, model.parameters(), args.lr, args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    best_val = math.inf
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_count += len(xb)

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 2 == 0:
            print(f"lstm/{optimizer_name} epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if patience_left <= 0:
            print(f"lstm/{optimizer_name} early stopped at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logits = predict_logits(model, x_test, args.batch_size, device)
    metric_result = topk_metrics(logits, y_test, args.top_k)
    topk_items = metric_result.pop("topk_items")

    return {
        "model": "lstm",
        "optimizer": optimizer_name,
        "best_val_loss": best_val,
        "history": pd.DataFrame(history),
        "metrics": metric_result,
        "topk_items": topk_items,
        "test_users": test_users,
        "test_times": test_times,
        "y_test": y_test,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_transactions(args.data_dir)
    df = keep_top_items(df, args.top_items)
    item_to_id, id_to_item = build_item_vocab(df)
    num_items = len(item_to_id)
    print(f"Transactions after cleaning: {len(df):,}")
    print(f"Customers: {df['CustomerID'].nunique():,}")
    print(f"Candidate items: {num_items:,}")

    samples = make_time_window_samples(
        df=df,
        item_to_id=item_to_id,
        history_hours=args.history_hours,
        future_hours=args.future_hours,
        stride_hours=args.stride_hours,
        max_seq_len=args.max_seq_len,
        min_history_events=args.min_history_events,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("No training samples were generated. Try increasing history/future windows or lowering min-history-events.")

    from collections import Counter

    target_counter = Counter()
    for sample in samples:
        target_counter.update(sample.target_items)
    print(f"Generated samples: {len(samples):,}")
    print(f"Most common future target item ids: {target_counter.most_common(5)}")

    x, y, user_ids, anchor_times = vectorize_samples(samples, args.max_seq_len, num_items)
    splits = chronological_split(x, y, user_ids, anchor_times)

    optimizer_names = [name.strip().lower() for name in args.optimizers.split(",")]
    all_metrics = []
    histories = {}

    for optimizer_name in optimizer_names:
        run_name = f"lstm_{optimizer_name}"
        print(f"\nRunning {run_name}")
        result = train_one_run(args, splits, num_items, optimizer_name, device)

        result["history"].to_csv(args.output_dir / f"history_{run_name}.csv", index=False)
        histories[run_name] = result["history"]

        rec_df = make_recommendation_frame(result, id_to_item, sample_count=200)
        rec_df.to_csv(args.output_dir / f"recommendations_{run_name}.csv", index=False, encoding="utf-8-sig")

        row = {
            "model": "lstm",
            "optimizer": optimizer_name,
            "best_val_loss": result["best_val_loss"],
        }
        row.update(result["metrics"])
        all_metrics.append(row)

    metrics_df = pd.DataFrame(all_metrics).sort_values([f"recall@{args.top_k}", f"ndcg@{args.top_k}"], ascending=False)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    plot_metric_bars(metrics_df, args.output_dir, args.top_k)
    plot_loss_curves(histories, args.output_dir)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
