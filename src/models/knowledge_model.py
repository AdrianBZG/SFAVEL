"""
Module defining the knowledge model
"""

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import RGCNConv, RGATConv

from src.models.scorer.l2_scorer import L2Scorer


class KnowledgeModel(nn.Module):
    def __init__(self, num_nodes, num_relations, gnn_type, layers, dropout=0.1, llm_name=None, llm_dim=256, device='cpu'):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.input_channels = llm_dim
        self.edge_dim = llm_dim
        self.hidden_channels = llm_dim
        self.llm_name = llm_name
        self.device = device
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.scorer = L2Scorer()

        if gnn_type == "rgcn":
            gnn_layer_class = RGCNConv
        elif gnn_type == "rgat":
            gnn_layer_class = RGATConv
        else:
            raise ValueError(f'Unrecognized GNN type {gnn_type} when creating Knowledge Model')

        self.layer_list = nn.ModuleList()
        self.convs = nn.ModuleList()

        for conv_inx, conv_dim in enumerate(layers):
            if conv_inx == 0:
                self.convs.append(gnn_layer_class(self.input_channels, conv_dim, num_relations))
                self.layer_list.append(nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(self.dropout)))
            else:
                self.convs.append(gnn_layer_class(layers[conv_inx-1], conv_dim, num_relations))
                self.layer_list.append(nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(self.dropout)))

        self.triplet_projector = Linear(layers[-1] * 2,
                                        self.input_channels)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type, edge_attr=None):
        for layer_id in range(len(self.convs)):
            if self.gnn_type == "rgat":
                x = self.convs[layer_id](x, edge_index, edge_type, edge_attr)
                x = self.layer_list[layer_id](x)
            elif self.gnn_type == "rgcn":
                x = self.convs[layer_id](x, edge_index, edge_type)
                x = self.layer_list[layer_id](x)

        # Get triplet embeddings projection
        triplet_embeddings = torch.cat([x[edge_index[0]],
                                        x[edge_index[1]]], dim=1)

        triplet_embeddings = self.triplet_projector(triplet_embeddings)
        return triplet_embeddings

    def score(self, z_claims, z_triplets):
        return self.scorer.score(z_claims, z_triplets)

    @torch.no_grad()
    def _get_graph_random_sample(self, kg, num_samples):
        rnd_index = torch.randint(low=0, high=kg.graph.edge_index.size(1),
                                  size=[num_samples], device=self.device)
        head_indices = kg.graph.edge_index[0][rnd_index]
        tail_indices = kg.graph.edge_index[1][rnd_index]
        edge_index = torch.tensor([head_indices.tolist(),
                                   tail_indices.tolist()], device=self.device)
        edge_type = kg.graph.edge_type[rnd_index]
        edge_attr = kg.graph.edge_attr[rnd_index]
        return edge_index, edge_type, edge_attr

    def negative_sample(self, kg, heads, rels, tails, edge_attr, negative_samples, negative_perturbation_ratio=0.5):
        node_attr = kg.graph.x
        negative_z = []
        for b_idx in range(len(heads)):
            num_perturbations = int(heads[b_idx].numel() * negative_perturbation_ratio)
            rnd_index = torch.randint(self.num_nodes, heads[b_idx].size(), device=self.device)  # Here k == heads[b_idx].size()
            head_index = heads[b_idx].clone()
            head_index[:num_perturbations] = rnd_index[:num_perturbations]
            tail_index = tails[b_idx].clone()
            tail_index[num_perturbations:] = rnd_index[num_perturbations:]
            edge_type = rels[b_idx].clone()
            edge_attr_perturb = edge_attr[b_idx]

            edge_index = torch.tensor([head_index.tolist(),
                                       tail_index.tolist()], device=self.device)

            # Perform random sampling
            rnd_edge_index, rnd_edge_type, rnd_edge_attr = self._get_graph_random_sample(kg, negative_samples)

            # Create new subgraph as concatenation of previous two
            edge_type = torch.cat([edge_type, rnd_edge_type], dim=0)
            edge_attr_perturb = torch.cat([edge_attr_perturb, rnd_edge_attr], dim=0)
            edge_index = torch.cat([edge_index, rnd_edge_index], dim=1)

            if self.gnn_type == "rgat":
                negative_z.append(self(x=node_attr,
                                       edge_index=edge_index,
                                       edge_type=edge_type,
                                       edge_attr=edge_attr_perturb))
            elif self.gnn_type == "rgcn":
                negative_z.append(self(x=node_attr,
                                       edge_index=edge_index,
                                       edge_type=edge_type))

        return torch.stack(negative_z)
