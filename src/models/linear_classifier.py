"""
Module defining a simple linear classifier to use in finetuning tasks
"""

import torch
import torch.nn as nn

from utils import get_lm_emb_dimensionality


class Classifier(nn.Module):
    def __init__(self, config, hidden_dim=512, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.input_dim = get_lm_emb_dimensionality(config["language_model"]) * 2
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(config["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(config["dropout"]),
            nn.Linear(self.hidden_dim, out_features=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.init_params()

    def forward(self, x_lm, triplets_z):
        x_lm, triplets_z = x_lm.to(self.device), triplets_z.to(self.device)

        if self.config["k"] > 1:
            triplets_z = triplets_z.mean(1)

        x = torch.cat([x_lm, triplets_z], dim=1)
        x = self.net(x)
        y_hat = self.sigmoid(x)
        return y_hat

    def calculate_loss(self, y_hat, y_true):
        if y_hat.shape != y_true.shape:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y_true)
        return loss

    def prepare_optimizers(self):
        main_params = list(self.parameters())
        model_parameters_count = sum(p.numel() for p in main_params if p.requires_grad)
        print(f"The model has {model_parameters_count} trainable parameters")

        optimizer = torch.optim.Adam(list(main_params),
                                     lr=self.config['learning_rate'])

        return optimizer

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
