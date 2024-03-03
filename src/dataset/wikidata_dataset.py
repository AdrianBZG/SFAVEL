"""
Module to wrap the Wikidata dataset into a PyTorch Dataset
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

logger = logging.getLogger("wikidata-dataset")


class Wikidata5mDataset(Dataset):
    def __init__(self, root_path="./data", node_feature_dim=128, **kwargs):
        self.triplets_file_path = f"{root_path}/preprocessed/wikidata_triplets.pkl"
        self.triplets_emb_file_path = f"{root_path}/preprocessed/wikidata_triplets_emb.pkl"
        self.node_feature_dim = node_feature_dim

        self.ent2id = {}
        self.rel2id = {}
        self.graph = None
        self._populate_data()

    def _populate_data(self):
        # Load entities and relations
        self._load_entities()
        self._load_relations()

        # Load graph data
        self._load_graph()

    def _load_graph(self):
        # Load triplets and embeddings from disk
        with open(self.triplets_file_path, "rb") as file:
            triplets_data = pickle.load(file)

        with open(self.triplets_emb_file_path, "rb") as file:
            triplets_embeddings = pickle.load(file)

        graph = nx.DiGraph()
        already_seen_edges = set()

        for triplet_data, triplet_embedding in tqdm(zip(triplets_data.values(), triplets_embeddings.values())):
            head, relation, tail = triplet_data["head"], triplet_data["relation"], triplet_data["tail"]
            triplet_verbalized = triplet_data["verbalized"]

            # Sanity check
            if head not in self.ent2id or tail not in self.ent2id:
                raise ValueError(f'An entity was not loaded correctly ({head} | {tail})')

            if relation not in self.rel2id:
                raise ValueError(f'A relation was not loaded correctly ({relation})')

            head_id, rel_id, tail_id = self.ent2id[head], self.rel2id[relation], self.ent2id[tail]

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
                               edge_attr=torch.rand(1, self.node_feature_dim)[0],
                               edge_name=relation,
                               triplet_verbalized=triplet_verbalized,
                               triplet_embedding=triplet_embedding)

                # Prevent duplicates and reverse edges
                already_seen_edges.add((head_id, tail_id, rel_id))
                already_seen_edges.add((tail_id, head_id, rel_id))

        graph = from_networkx(graph)
        graph.num_nodes = graph.x.shape[0]
        graph.num_edge_types = len(self.rel2id)
        self.graph = graph

    def _load_entities(self):
        ent2id = {}
        with open(self.triplets_file_path, "rb") as file:
            triplets_data = pickle.load(file)
            heads = [triplet["head"] for triplet in triplets_data.values()]
            tails = [triplet["tail"] for triplet in triplets_data.values()]

        for entity in heads + tails:
            entity_id = len(ent2id)
            ent2id[entity] = entity_id

        self.ent2id = ent2id

    def _load_relations(self):
        rel2id = {}
        with open(self.triplets_file_path, "rb") as file:
            triplets_data = pickle.load(file)
            all_relations = [triplet["relation"] for triplet in triplets_data.values()]

        for relation in all_relations:
            rel_id = len(rel2id)
            rel2id[relation] = rel_id

        self.rel2id = rel2id

    def __getitem__(self, idx):
        '''
        Get a single triplet by index
        '''
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
        return self.graph.num_nodes
