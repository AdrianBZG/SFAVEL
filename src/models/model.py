"""
Module defining the main SFAVEL model
"""

import torch
import torch.nn as nn

from src.models.knowledge_model import KnowledgeModel
from src.models.losses import DistillationLoss, IntraSampleLoss, ScoringLoss


class SFAVEL(torch.nn.Module):
    def __init__(self, config, num_nodes, num_relations, lm_dim, device='cpu'):
        super().__init__()
        self.config = config
        self.lm_dim = lm_dim
        self.device = device
        self.temperature = config["temperature"]

        self.knowledge_model = KnowledgeModel(num_nodes=num_nodes,
                                              num_relations=num_relations,
                                              llm_name=config["language_model"],
                                              gnn_type=config["km_type"],
                                              layers=config["km_layers"],
                                              dropout=config["dropout"],
                                              device=device,
                                              llm_dim=lm_dim)

        self.distillation_loss = DistillationLoss()
        self.intra_sample_loss = IntraSampleLoss(temperature=self.temperature)
        self.scoring_loss = ScoringLoss(margin=self.config["scoring_loss_margin"])

        self.init_params()

    def _get_top_k_triplets(self, kg, x_lm, x_triplets, k=10):
        top_k_scored_triplets = self.knowledge_model.score(x_lm, x_triplets).topk(k,
                                                                                  largest=True).indices

        candidates_src = torch.stack(
            [torch.index_select(kg.graph.edge_index[0], 0, indices) for indices in top_k_scored_triplets])
        candidates_dst = torch.stack(
            [torch.index_select(kg.graph.edge_index[1], 0, indices) for indices in top_k_scored_triplets])
        candidates_edge_type = torch.stack(
            [torch.index_select(kg.graph.edge_type, 0, indices) for indices in top_k_scored_triplets])
        candidates_edge_attr = torch.stack(
            [torch.index_select(kg.graph.edge_attr, 0, indices) for indices in top_k_scored_triplets])
        candidates_z = torch.stack([torch.index_select(x_triplets, 0, indices) for indices in top_k_scored_triplets])
        candidates_lm_z = torch.stack([torch.index_select(kg.graph.triplet_embedding, 0, indices) for indices in top_k_scored_triplets])

        return candidates_z, candidates_lm_z, candidates_src, candidates_dst, candidates_edge_type, candidates_edge_attr

    def forward(self, batch, kg, k=10):
        x_lm = torch.stack([claim["embedding"] for claim in batch]).to(self.device)

        kg_triplets_z = self.knowledge_model(x=kg.graph.x,
                                             edge_index=kg.graph.edge_index,
                                             edge_type=kg.graph.edge_type,
                                             edge_attr=kg.graph.edge_attr)

        candidates_z, candidates_lm_z, candidates_src, candidates_dst, candidates_edge_type, candidates_edge_attr = self._get_top_k_triplets(kg, x_lm, kg_triplets_z, k)

        # Sample in-batch negative indices
        with torch.no_grad():
            negatives_z = self.knowledge_model.negative_sample(kg=kg,
                                                               heads=candidates_src,
                                                               rels=candidates_edge_type,
                                                               tails=candidates_dst,
                                                               edge_attr=candidates_edge_attr,
                                                               negative_samples=self.config["negative_samples"],
                                                               negative_perturbation_ratio=self.config["negative_perturbation_ratio"])

        # Score positives and negatives
        pos_scores = self.knowledge_model.score(x_lm, candidates_z)
        neg_scores = self.knowledge_model.score(x_lm, negatives_z)
        return x_lm, pos_scores, neg_scores, candidates_z, candidates_lm_z, negatives_z

    @torch.no_grad()
    def inference(self, batch, kg, k=10):
        x_lm = torch.stack([claim["embedding"] for claim in batch]).to(self.device)

        kg_triplets_z = self.knowledge_model(x=kg.graph.x,
                                             edge_index=kg.graph.edge_index,
                                             edge_type=kg.graph.edge_type,
                                             edge_attr=kg.graph.edge_attr)

        candidates_z, candidates_lm_z, candidates_src, candidates_dst, candidates_edge_type, candidates_edge_attr = self._get_top_k_triplets(kg, x_lm, kg_triplets_z, k)

        return x_lm, candidates_z

    def calculate_loss(self, x_lm, pos_scores, neg_scores, candidates_z, candidates_lm_z, negatives_z):
        loss = 0

        # Claim-Triplet distillation loss
        distillation_loss = self.distillation_loss(x_lm, candidates_lm_z) * self.config["loss_alpha"]["distillation"]
        loss += distillation_loss

        # Intra-Sample contrastive loss
        intra_sample_loss = self.intra_sample_loss(x_lm, candidates_z, negatives_z) * self.config["loss_alpha"]["intra_sample"]
        loss += intra_sample_loss

        # Scoring loss
        scoring_loss = self.scoring_loss(pos_scores, neg_scores) * self.config["loss_alpha"]["scoring"]
        loss += scoring_loss

        return {"loss": loss,
                "distillation_loss": distillation_loss,
                "intra_sample_loss": intra_sample_loss,
                "scoring_loss": scoring_loss}

    def prepare_optimizers(self):
        main_params = list(self.parameters())
        model_parameters_count = sum(p.numel() for p in main_params if p.requires_grad)
        print(f"The model has {model_parameters_count} trainable parameters")

        optimizers = []

        optimizer_type = self.config['optimizer']
        if optimizer_type == "adamw":
            net_optimizer = torch.optim.AdamW(list(main_params),
                                              lr=self.config['learning_rate'],
                                              weight_decay=self.config['weight_decay'])
        elif optimizer_type == "adam":
            net_optimizer = torch.optim.Adam(list(main_params),
                                             lr=self.config['learning_rate'])
        elif optimizer_type == "sgd":
            net_optimizer = torch.optim.SGD(list(main_params),
                                            lr=self.config['learning_rate'])
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        optimizers.append(net_optimizer)
        return optimizers

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
