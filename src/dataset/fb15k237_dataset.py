"""
Module to wrap the FB15K-237 dataset into a PyTorch Dataset
"""

import logging
import pickle
import networkx as nx
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("fb15k237-dataset")


class FB15K237Dataset(Dataset):
    def __init__(self, root_path="./data", split="train", as_graph=False, node_feature_dim=128, **kwargs):
        assert split in {"train", "dev", "test"}
        self.data_file_path = f"{root_path}/preprocessed/fb15k237_{split}_data.pkl"
        self.emb_file_path = f"{root_path}/preprocessed/fb15k237_{split}_embeddings.pkl"
        self.as_graph = as_graph
        self.node_feature_dim = node_feature_dim

        self.heads, self.relations, self.tails, self.verbalized, self.embeddings = [], [], [], [], []
        self.graph = None
        self._populate_data()

        # Safety checks
        assert len(self.heads) == len(self.relations)
        assert len(self.heads) == len(self.tails)
        assert len(self.heads) == len(self.verbalized)
        assert len(self.heads) == len(self.embeddings)
        if as_graph and not self.graph:
            raise ValueError(f'The graph has not been initialized')

    def _populate_data(self):
        with open(self.data_file_path, "rb") as file:
            data = pickle.load(file)

        with open(self.emb_file_path, "rb") as file:
            embeddings = pickle.load(file)

        for triplet_data, triplet_emb in zip(data.values(), embeddings.values()):
            head_text = triplet_data["head"]
            rel_text = triplet_data["relation"]
            tail_text = triplet_data["tail"]
            triplet_verbalized = triplet_data["verbalized"]
            self.heads.append(head_text)
            self.relations.append(rel_text)
            self.tails.append(tail_text)
            self.verbalized.append(triplet_verbalized)
            self.embeddings.append(triplet_emb)

        if self.as_graph:
            self._load_graph()

    def _load_graph(self):
        graph = nx.DiGraph()

        ent2id, rel2id = {}, {}
        already_seen_edges = set()

        for head, relation, tail, triplet_verbalized, triplet_embedding in tqdm(zip(self.heads, self.relations, self.tails, self.verbalized, self.embeddings)):
            if head not in ent2id :
                ent2id[head] = len(ent2id)

            if tail not in ent2id:
                ent2id[tail] = len(ent2id)

            if relation not in rel2id:
                rel2id[relation] = len(rel2id)

            head_id, rel_id, tail_id = ent2id[head], rel2id[relation], ent2id[tail]

            if head_id not in graph:
                graph.add_node(head_id,
                               x=torch.rand(1, self.node_feature_dim)[0],
                               node_name=head)

            if tail_id not in graph:
                graph.add_node(tail_id,
                               x=torch.rand(1, self.node_feature_dim)[0],
                               node_name=tail)

            if (head_id, tail_id, rel_id) not in already_seen_edges:
                graph.add_edge(head_id, tail_id,
                               edge_type=rel_id,
                               edge_name=relation,
                               triplet_verbalized=triplet_verbalized,
                               triplet_embedding=triplet_embedding)

                # Prevent duplicates and reverse edges
                already_seen_edges.add((head_id, tail_id, rel_id))
                already_seen_edges.add((tail_id, head_id, rel_id))

        graph = from_networkx(graph)
        graph.num_nodes = graph.x.shape[0]
        graph.num_edge_types = len(rel2id)
        self.graph = graph

    def __getitem__(self, idx):
        if not self.as_graph:
            # Used as fact dataset, return single element
            return {"head": self.heads[idx],
                    "relation": self.relations[idx],
                    "tail": self.tails[idx],
                    "verbalized": self.verbalized[idx],
                    "embedding": torch.tensor(self.embeddings[idx])}
        else:
            # Used as knowledge base, return single triplet
            if idx >= len(self.graph.triplet_verbalized):
                raise ValueError(f"Requested triplet with id {idx} but {len(self.graph.triplet_verbalized)} available")

            head = self.graph.edge_index[0][idx]
            head_name = self.graph.node_name[head]
            tail = self.graph.edge_index[1][idx]
            tail_name = self.graph.node_name[tail]
            relation = self.graph.edge_type[idx]
            relation_name = self.graph.edge_name[idx]
            verbalized = self.graph.triplet_verbalized[idx]
            embedding = self.graph.triplet_embedding[idx]

            return {"head": head,
                    "head_name": head_name,
                    "relation": relation,
                    "relation_name": relation_name,
                    "tail": tail,
                    "tail_name": tail_name,
                    "verbalized": verbalized,
                    "embedding": torch.tensor(embedding)}

    def __len__(self):
        if not self.as_graph:
            return len(self.verbalized)
        else:
            return self.graph.num_nodes
