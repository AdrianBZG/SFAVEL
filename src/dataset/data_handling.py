"""
Module to define utility functions to obtain dataset and data loaders objects
"""
import logging
import time
import os
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.dataset.fever_dataset import FEVERDataset
from src.dataset.fb15k237_dataset import FB15K237Dataset
from src.dataset.wikidata_dataset import Wikidata5mDataset


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("data-handling")


def batch_collate_function(fact_batch):
    return fact_batch


def get_fact_dataset(root_path, split, fact_dataset):
    if fact_dataset == "FEVER":
        fact_dataset_class = FEVERDataset
    elif fact_dataset == "FB15K-237":
        fact_dataset_class = FB15K237Dataset
    else:
        raise ValueError(f"Unknown fact dataset: {fact_dataset}")

    logger.info(f"Loading fact dataset ({fact_dataset})...")
    fact_dataset = fact_dataset_class(root_path=root_path, split=split)
    logger.info(f"Loaded fact dataset with size {len(fact_dataset)}")

    return fact_dataset


def get_kg_dataset(root_path, kg_dataset, node_feature_dim):
    if kg_dataset == "wikidata5m":
        kg_dataset_class = Wikidata5mDataset
    elif kg_dataset == "FB15K-237":
        kg_dataset_class = FB15K237Dataset
    else:
        raise ValueError(f"Unknown fact dataset: {kg_dataset}")

    logger.info(f"Loading knowledge graph dataset ({kg_dataset})...")
    kg_dataset = kg_dataset_class(root_path=root_path, node_feature_dim=node_feature_dim, as_graph=True)
    logger.info(f"Loaded knowledge graph dataset with size {len(kg_dataset)}")

    return kg_dataset


def get_dataset(root_path="./data", split="train", fact_dataset="FEVER", kg_dataset="wikidata5m",
                node_feature_dim=128):
    assert split in {"train", "dev", "test"}

    logger.info(f"Loading dataset for split {split}")
    start_time = time.time()

    fact_dataset = get_fact_dataset(root_path, split, fact_dataset)
    kg_dataset = get_kg_dataset(root_path, kg_dataset, node_feature_dim=node_feature_dim)

    logger.info(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds. "
                f"Dataset size: {len(fact_dataset)}")

    return fact_dataset, kg_dataset


def get_data_loader(fact_dataset, kg_dataset, batch_size=16, shuffle=False, num_workers=4):
    data_loader = DataLoader(fact_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=lambda batch: batch_collate_function(batch))
    return data_loader
