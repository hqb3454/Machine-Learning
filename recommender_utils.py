from __future__ import annotations

import math
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
PAD_ID = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = data_dir / "Online Retail.xlsx"
    if xlsx_path.exists():
        return xlsx_path

    print(f"Downloading UCI Online Retail dataset from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, xlsx_path)
    return xlsx_path


def load_transactions(data_dir: Path) -> pd.DataFrame:
    xlsx_path = download_dataset(data_dir)
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    df = df.dropna(subset=["CustomerID", "StockCode", "InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)
    return df


def keep_top_items(df: pd.DataFrame, top_items: int) -> pd.DataFrame:
    item_counts = df["StockCode"].value_counts()
    selected_items = set(item_counts.head(top_items).index)
    return df[df["StockCode"].isin(selected_items)].copy()


def build_item_vocab(df: pd.DataFrame):
    items = sorted(df["StockCode"].unique())
    item_to_id = {item: idx + 1 for idx, item in enumerate(items)}
    id_to_item = {idx: item for item, idx in item_to_id.items()}
    return item_to_id, id_to_item


@dataclass
class WindowSample:
    user_id: str
    anchor_time: pd.Timestamp
    history_items: list[int]
    target_items: list[int]


def make_time_window_samples(
    df: pd.DataFrame,
    item_to_id: dict[str, int],
    history_hours: int,
    future_hours: int,
    stride_hours: int,
    max_seq_len: int,
    min_history_events: int,
    max_samples: int | None,
) -> list[WindowSample]:
    samples: list[WindowSample] = []
    history_delta = pd.Timedelta(hours=history_hours)
    future_delta = pd.Timedelta(hours=future_hours)
    stride_delta = pd.Timedelta(hours=stride_hours)

    for user_id, user_df in df.groupby("CustomerID", sort=False):
        user_df = user_df.sort_values("InvoiceDate")
        if len(user_df) < min_history_events + 1:
            continue

        start_time = user_df["InvoiceDate"].min() + history_delta
        end_time = user_df["InvoiceDate"].max() - future_delta
        if start_time > end_time:
            continue

        anchor_time = start_time
        while anchor_time <= end_time:
            hist_start = anchor_time - history_delta
            fut_end = anchor_time + future_delta

            hist_df = user_df[(user_df["InvoiceDate"] >= hist_start) & (user_df["InvoiceDate"] < anchor_time)]
            fut_df = user_df[(user_df["InvoiceDate"] >= anchor_time) & (user_df["InvoiceDate"] < fut_end)]

            if len(hist_df) >= min_history_events and len(fut_df) > 0:
                history_items = [item_to_id[x] for x in hist_df["StockCode"].tolist() if x in item_to_id]
                target_items = sorted({item_to_id[x] for x in fut_df["StockCode"].tolist() if x in item_to_id})
                if history_items and target_items:
                    samples.append(
                        WindowSample(
                            user_id=user_id,
                            anchor_time=anchor_time,
                            history_items=history_items[-max_seq_len:],
                            target_items=target_items,
                        )
                    )

            if max_samples is not None and len(samples) >= max_samples:
                return samples

            anchor_time += stride_delta

    return samples


def vectorize_samples(samples: list[WindowSample], max_seq_len: int, num_items: int):
    n = len(samples)
    x = np.zeros((n, max_seq_len), dtype=np.int64)
    y = np.zeros((n, num_items + 1), dtype=np.float32)
    user_ids = []
    anchor_times = []

    for i, sample in enumerate(samples):
        seq = sample.history_items[-max_seq_len:]
        x[i, -len(seq) :] = seq
        for item_id in sample.target_items:
            y[i, item_id] = 1.0
        user_ids.append(sample.user_id)
        anchor_times.append(sample.anchor_time)

    return x, y, np.asarray(user_ids), np.asarray(anchor_times)


def chronological_split(x, y, user_ids, anchor_times, train_ratio=0.7, val_ratio=0.15):
    order = np.argsort(anchor_times)
    x, y, user_ids, anchor_times = x[order], y[order], user_ids[order], anchor_times[order]

    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": (x[:train_end], y[:train_end], user_ids[:train_end], anchor_times[:train_end]),
        "val": (x[train_end:val_end], y[train_end:val_end], user_ids[train_end:val_end], anchor_times[train_end:val_end]),
        "test": (x[val_end:], y[val_end:], user_ids[val_end:], anchor_times[val_end:]),
    }


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_loader(x, y, batch_size: int, shuffle: bool):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            total_count += len(xb)
    return total_loss / max(total_count, 1)


def topk_metrics(logits: np.ndarray, targets: np.ndarray, k: int):
    k = min(k, logits.shape[1])
    if k <= 0:
        raise ValueError("top_k must be positive.")
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    sorted_topk = np.take_along_axis(
        topk,
        np.argsort(-np.take_along_axis(logits, topk, axis=1), axis=1),
        axis=1,
    )

    precision_scores = []
    recall_scores = []
    hit_scores = []
    ndcg_scores = []
    map_scores = []

    for pred_items, target_row in zip(sorted_topk, targets):
        true_items = set(np.flatnonzero(target_row > 0.5).tolist())
        true_items.discard(PAD_ID)
        if not true_items:
            continue

        hits = [1 if item in true_items else 0 for item in pred_items]
        hit_count = sum(hits)

        precision_scores.append(hit_count / k)
        recall_scores.append(hit_count / len(true_items))
        hit_scores.append(1.0 if hit_count > 0 else 0.0)

        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = min(len(true_items), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        running_hits = 0
        precision_at_hits = []
        for rank, hit in enumerate(hits, start=1):
            if hit:
                running_hits += 1
                precision_at_hits.append(running_hits / rank)
        map_scores.append(sum(precision_at_hits) / min(len(true_items), k) if precision_at_hits else 0.0)

    return {
        f"precision@{k}": float(np.mean(precision_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        f"hit_rate@{k}": float(np.mean(hit_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"map@{k}": float(np.mean(map_scores)),
        "topk_items": sorted_topk,
    }


def predict_logits(model, x, batch_size: int, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(outputs)


def make_recommendation_frame(result, id_to_item: dict[int, str], sample_count: int):
    rows = []
    n = min(sample_count, len(result["test_users"]))
    for i in range(n):
        pred_item_ids = result["topk_items"][i].tolist()
        true_item_ids = np.flatnonzero(result["y_test"][i] > 0.5).tolist()
        true_item_ids = [item_id for item_id in true_item_ids if item_id != PAD_ID]
        rows.append(
            {
                "user_id": result["test_users"][i],
                "anchor_time": result["test_times"][i],
                "recommended_items": ",".join(id_to_item[item_id] for item_id in pred_item_ids if item_id in id_to_item),
                "true_future_items": ",".join(id_to_item[item_id] for item_id in true_item_ids if item_id in id_to_item),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(metrics_df: pd.DataFrame, output_dir: Path, top_k: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = metrics_df["model"] + "+" + metrics_df["optimizer"]
    x = np.arange(len(labels))

    plt.figure(figsize=(max(10, len(labels) * 0.8), 5))
    plt.bar(x - 0.2, metrics_df[f"recall@{top_k}"], width=0.4, label=f"Recall@{top_k}")
    plt.bar(x + 0.2, metrics_df[f"ndcg@{top_k}"], width=0.4, label=f"NDCG@{top_k}")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Recommendation Ranking Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_recall_ndcg.png", dpi=160)
    plt.close()

    plt.figure(figsize=(max(10, len(labels) * 0.8), 5))
    plt.bar(x - 0.2, metrics_df[f"precision@{top_k}"], width=0.4, label=f"Precision@{top_k}")
    plt.bar(x + 0.2, metrics_df[f"hit_rate@{top_k}"], width=0.4, label=f"HitRate@{top_k}")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Recommendation Hit Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_precision_hit_rate.png", dpi=160)
    plt.close()


def plot_loss_curves(histories: dict[str, pd.DataFrame], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for run_name, history in histories.items():
        if history.empty:
            continue
        plt.plot(history["epoch"], history["val_loss"], label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss Curves")
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close()
