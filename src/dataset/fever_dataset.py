"""
Module to wrap the Fever dataset into a PyTorch Dataset
"""

import logging
import pickle
import torch
from torch.utils.data import Dataset


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("fever-dataset")


class FEVERDataset(Dataset):
    def __init__(self, root_path="./data", split="train"):
        assert split in {"train", "dev", "test"}
        self.data_file_path = f"{root_path}/preprocessed/fever_{split}_data.pkl"
        self.emb_file_path = f"{root_path}/preprocessed/fever_{split}_embeddings.pkl"

        self.ids, self.labels, self.claims, self.embeddings = [], [], [], []
        self._populate_data()

        # Safety checks
        assert len(self.ids) == len(self.labels)
        assert len(self.ids) == len(self.claims)
        assert len(self.ids) == len(self.embeddings)

    def _populate_data(self):
        with open(self.data_file_path, "rb") as file:
            data = pickle.load(file)

        with open(self.emb_file_path, "rb") as file:
            embeddings = pickle.load(file)

        for claim_id, claim_data in data.items():
            claim_text = claim_data["claim"]
            claim_label = claim_data["label"]
            claim_emb = embeddings[claim_id]
            self.ids.append(claim_id)
            self.claims.append(claim_text)
            self.labels.append(claim_label)
            self.embeddings.append(claim_emb)

    def __getitem__(self, idx):
        return {"id": torch.tensor(self.ids[idx]),
                "label": torch.tensor(1 if self.labels[idx] == "SUPPORTS" else 0, dtype=torch.float),
                "claim": self.claims[idx],
                "embedding": torch.tensor(self.embeddings[idx])}

    def __len__(self):
        return len(self.ids)
