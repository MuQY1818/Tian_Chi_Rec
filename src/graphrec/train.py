import argparse
import os
from typing import List
import torch
from torch.optim import Adam
from tqdm import trange

from data import GraphDataset
from model import LightGCN


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_hr_recall(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    dataset: GraphDataset,
    k: int = 10,
) -> float:
    """Evaluate hit rate@k on leave-one-out validation (only for valid users)."""
    users = dataset.get_eval_users()
    if len(users) == 0:
        return 0.0

    # Collect all training items
    train_items = set()
    for items in dataset.user_pos_train.values():
        train_items.update(items)

    hits = 0
    valid_users = 0
    for u in users:
        pos_item = next(iter(dataset.user_pos_val[u]))
        # Skip users with cold-start items (not in training)
        if pos_item not in train_items:
            continue
        valid_users += 1

        seen = dataset.get_user_seen_items(u)
        u_vec = user_emb[u]
        scores = (item_emb @ u_vec)
        if len(seen) > 0:
            scores[list(seen)] = -1e9
        topk = torch.topk(scores, k=k).indices.tolist()
        if pos_item in topk:
            hits += 1

    if valid_users == 0:
        return 0.0
    print(f"[DEBUG] Valid eval users: {valid_users}/{len(users)} ({valid_users/len(users)*100:.1f}%)")
    return hits / valid_users


def generate_submission(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    dataset: GraphDataset,
    topk: int = 5,
    path: str = "submission.txt",
):
    with open(path, "w", encoding="utf-8") as f:
        for u in range(dataset.num_users):
            u_vec = user_emb[u]
            scores = (item_emb @ u_vec)
            seen = dataset.get_user_seen_items(u)
            if len(seen) > 0:
                scores[list(seen)] = -1e9
            recs = torch.topk(scores, k=topk).indices.tolist()
            uid = dataset.id2user[u]
            for iid in recs:
                item_id = dataset.id2item[iid]
                f.write(f"{uid}\t{item_id}\n")


def main():
    parser = argparse.ArgumentParser(description="LightGCN graph recommendation (compact pipeline)")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--graph_dir", type=str, default="")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset
    dataset = GraphDataset(data_dir=args.data_dir, sample_ratio=args.sample_ratio, min_interactions=1, seed=args.seed)
    if args.graph_dir and os.path.exists(os.path.join(args.graph_dir, "train_edges.pt")):
        print(f"Loading prepared graph from {args.graph_dir}")
        dataset.load_from_prepared(args.graph_dir)
    else:
        dataset.load()

    print(
        f"Users={dataset.num_users}, Items={dataset.num_items}, Edges(train)={dataset.edge_index.size(1)//2}"
    )

    # Model
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        edge_index=dataset.edge_index.to(args.device),
        embed_dim=args.embed_dim,
        n_layers=args.layers,
    ).to(args.device)

    optim = Adam(model.parameters(), lr=args.lr)

    # Training loop
    steps_per_epoch = max(1, dataset.num_users // max(1, args.batch_size))
    print(f"\n🚀 Starting training with {steps_per_epoch} steps per epoch")
    print(f"📊 Model: {args.layers}-layer LightGCN, {args.embed_dim}D embeddings")
    print(f"⚙️  Config: lr={args.lr}, batch_size={args.batch_size}, device={args.device}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        # Training progress bar
        pbar = trange(steps_per_epoch, desc=f"🔥 Epoch {epoch:2d}/{args.epochs}",
                      unit="batch", ncols=100, leave=True)

        for step in pbar:
            u, pi, ni = dataset.sample_batch(args.batch_size)
            if u.numel() == 0:
                continue
            u = u.to(args.device)
            pi = pi.to(args.device)
            ni = ni.to(args.device)

            user_emb, item_emb = model()
            u_e = user_emb[u]
            pi_e = item_emb[pi]
            ni_e = item_emb[ni]

            loss = LightGCN.bpr_loss(u_e, pi_e, ni_e, l2_reg=args.l2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

            # Update progress bar with current loss
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Eval
        print(f"📈 Evaluating epoch {epoch}...")
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model()
            hr = evaluate_hr_recall(user_emb, item_emb, dataset, k=max(10, args.topk))

        avg_loss = total_loss/steps_per_epoch
        print(f"✅ Epoch {epoch:2d}: loss={avg_loss:.4f} HR@{max(10, args.topk)}={hr:.4f}")

        # Progress indicator
        progress = epoch / args.epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"📊 Progress: [{bar}] {progress*100:.1f}%")
        print("-" * 60)

    # Export submission
    print(f"🎯 Generating final recommendations...")
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()
    generate_submission(user_emb, item_emb, dataset, topk=args.topk, path="submission.txt")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training completed successfully!")
    print(f"📝 Recommendations saved to: submission.txt")
    print(f"👥 Total users: {dataset.num_users:,}")
    print(f"🛒 Total items: {dataset.num_items:,}")
    print(f"⭐ Top-{args.topk} recommendations per user")
    print("=" * 60)


if __name__ == "__main__":
    main()
