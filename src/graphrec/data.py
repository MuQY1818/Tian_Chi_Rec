import os
import json
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import torch


class GraphDataset:
    """Builds a user–item bipartite graph for LightGCN.

    Loading priority:
    1) user_data_processed.pkl (if exists) with columns including user_id, item_id, datetime/time, behavior_type
    2) Raw text at {data_dir}/tianchi_fresh_comp_train_user_online_partA.txt (tab-separated)

    Splitting strategy: leave-one-out per user by time (last interaction → validation).
    Implicit feedback: any behavior_type (1/2/3/4) counts as interaction.
    """

    def __init__(
        self,
        data_dir: str = "dataset",
        sample_ratio: float = 1.0,
        min_interactions: int = 1,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.sample_ratio = sample_ratio
        self.min_interactions = min_interactions
        self.rng = np.random.default_rng(seed)

        self.user2id: Dict[int, int] = {}
        self.item2id: Dict[int, int] = {}
        self.id2user: List[int] = []
        self.id2item: List[int] = []

        self.num_users: int = 0
        self.num_items: int = 0
        self.num_nodes: int = 0

        self.user_pos_train: Dict[int, Set[int]] = {}
        self.user_pos_val: Dict[int, Set[int]] = {}

        self.edge_index: torch.Tensor = torch.empty(2, 0, dtype=torch.long)

    def _load_dataframe(self) -> pd.DataFrame:
        if os.path.exists("user_data_processed.pkl"):
            df = pd.read_pickle("user_data_processed.pkl")
            # Ensure datetime
            if "datetime" not in df.columns and "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
            return df

        # Fallback to raw file (use small dataset for testing)
        user_file = os.path.join(
            self.data_dir, "small_dataset.txt"
        )
        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
        # Read with optional sampling by rows (approximate)
        df = pd.read_csv(user_file, sep="\t", names=columns)
        if 0 < self.sample_ratio < 1.0:
            df = df.sample(frac=self.sample_ratio, random_state=2024)

        df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
        return df

    def _build_mappings(self, df: pd.DataFrame) -> None:
        users = df["user_id"].astype(int).unique()
        items = df["item_id"].astype(int).unique()
        users.sort()
        items.sort()
        self.user2id = {u: i for i, u in enumerate(users)}
        self.item2id = {it: i for i, it in enumerate(items)}
        self.id2user = list(users)
        self.id2item = list(items)
        self.num_users = len(users)
        self.num_items = len(items)
        self.num_nodes = self.num_users + self.num_items

    def _filter_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep valid records and implicit interactions
        df = df.dropna(subset=["user_id", "item_id", "datetime"])  # robust
        df = df.astype({"user_id": int, "item_id": int})
        if "behavior_type" in df.columns:
            df = df[df["behavior_type"].isin([1, 2, 3, 4])]
        df = df.sort_values(["user_id", "datetime"]).reset_index(drop=True)

        # Ensure minimum interactions per user
        user_counts = df.groupby("user_id").size()
        keep_users = user_counts[user_counts >= max(self.min_interactions, 1)].index
        df = df[df["user_id"].isin(keep_users)]
        return df

    def load(self) -> None:
        df = self._load_dataframe()
        df = self._filter_and_sort(df)
        self._build_mappings(df)

        # Build interactions with time
        df["uid"] = df["user_id"].map(self.user2id)
        df["iid"] = df["item_id"].map(self.item2id)

        # Deduplicate by (uid, iid, datetime) keeping earliest per hour to reduce redundancy
        df = df.drop_duplicates(subset=["uid", "iid", "datetime"]).reset_index(drop=True)

        # Leave-one-out split
        self.user_pos_train = {}
        self.user_pos_val = {}
        for uid, g in df.groupby("uid"):
            items = g.sort_values("datetime")["iid"].tolist()
            if len(items) == 1:
                # no validation for singletons → keep in train
                self.user_pos_train[uid] = set(items)
                continue
            self.user_pos_train[uid] = set(items[:-1])
            self.user_pos_val[uid] = {items[-1]}

        # Construct undirected edge_index (user ↔ item) using train set only
        src = []
        dst = []
        for u, items in self.user_pos_train.items():
            for i in items:
                ui = u
                ii = self.num_users + i  # offset item ids
                src.append(ui); dst.append(ii)
                src.append(ii); dst.append(ui)
        if len(src) == 0:
            self.edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            self.edge_index = torch.tensor([src, dst], dtype=torch.long)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        users = list(self.user_pos_train.keys())
        if len(users) == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        batch_users = self.rng.choice(users, size=min(batch_size, len(users)), replace=True)

        pos = []
        neg = []
        for u in batch_users:
            pos_item = self.rng.choice(list(self.user_pos_train[u]))
            # negative sampling until hit an unobserved item
            while True:
                cand = int(self.rng.integers(0, self.num_items))
                if cand not in self.user_pos_train[u]:
                    break
            pos.append(pos_item)
            neg.append(cand)

        u_t = torch.as_tensor(batch_users, dtype=torch.long)
        pi_t = torch.as_tensor(pos, dtype=torch.long)
        ni_t = torch.as_tensor(neg, dtype=torch.long)
        return u_t, pi_t, ni_t

    def get_eval_users(self) -> List[int]:
        return [u for u in self.user_pos_val.keys() if len(self.user_pos_val[u]) > 0]

    def get_user_seen_items(self, u: int) -> Set[int]:
        return self.user_pos_train.get(u, set())

    # Load from pre-prepared artifacts created by prepare.py
    def load_from_prepared(self, graph_dir: str) -> None:
        # mappings
        with open(os.path.join(graph_dir, "mappings.json"), "r", encoding="utf-8") as f:
            mp = json.load(f)
        self.num_users = int(mp["num_users"])
        self.num_items = int(mp["num_items"])
        self.num_nodes = self.num_users + self.num_items

        # Build reverse mappings
        self.user2id = {int(k): int(v) for k, v in mp["user2id"].items()}
        self.item2id = {int(k): int(v) for k, v in mp["item2id"].items()}
        self.id2user = [0] * self.num_users
        self.id2item = [0] * self.num_items
        for user_id, uid in self.user2id.items():
            self.id2user[uid] = user_id
        for item_id, iid in self.item2id.items():
            self.id2item[iid] = item_id

        # seen mapping
        with open(os.path.join(graph_dir, "seen_train.json"), "r", encoding="utf-8") as f:
            seen = json.load(f)
        self.user_pos_train = {int(k): set(map(int, v)) for k, v in seen.items()}

        # val pairs
        import pandas as pd
        vp = pd.read_parquet(os.path.join(graph_dir, "val_pairs.parquet"))
        self.user_pos_val = {}
        for row in vp.itertuples(index=False):
            self.user_pos_val.setdefault(int(row.uid), set()).add(int(row.iid_pos))

        # edges
        state = torch.load(os.path.join(graph_dir, "train_edges.pt"), map_location="cpu")
        self.edge_index = state["edge_index"].long()
