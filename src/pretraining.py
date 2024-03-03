"""
Main pretrain entrypoint
"""

import argparse
import os
import sys
import logging
import torch
from torch import multiprocessing as torchmultiprocessing
torchmultiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm, trange
import wandb
from dataset.data_handling import get_dataset, get_data_loader
from models.model import SFAVEL
from utils import set_seed, load_params_from_file, get_lm_emb_dimensionality, get_available_device

logging.basicConfig(level=logging.INFO,
                    format='[pretraining:%(levelname)s] %(message)s')


def run_epoch(dataloader, model, optimizers, kg, config, epoch=None):
    model.train()
    main_optimizer = optimizers[0]

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        x_lm, pos_scores, neg_scores, candidates_z, candidates_lm_z, negatives_z = model(batch, kg, k=config["k"])

        losses = model.calculate_loss(x_lm, pos_scores, neg_scores, candidates_z, candidates_lm_z, negatives_z)

        loss = losses["loss"]

        main_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        main_optimizer.step()

        log_metrics = {"train_loss": loss.item(),
                       "train_scoring_loss": losses["scoring_loss"].item(),
                       "train_distillation_loss": losses["distillation_loss"].item(),
                       "train_intra_sample_loss": losses["intra_sample_loss"].item(),
                       "train_epoch": epoch + 1,
                       "train_batch": batch_idx}

        wandb.log(log_metrics)
        logging.info(f'Epoch [{epoch + 1}] - Metrics: {log_metrics}')


def run_training(config):
    train_dataset, kg_dataset = get_dataset(root_path=config["root_path"],
                                            split="train",
                                            fact_dataset=config["fact_dataset"],
                                            kg_dataset=config["kg_dataset"],
                                            node_feature_dim=get_lm_emb_dimensionality(config["language_model"]))

    train_dataloader = get_data_loader(fact_dataset=train_dataset,
                                       kg_dataset=kg_dataset,
                                       batch_size=config["batch_size"],
                                       shuffle=True,
                                       num_workers=0)

    model = SFAVEL(config, len(kg_dataset), kg_dataset.graph.num_edge_types,
                   lm_dim=get_lm_emb_dimensionality(config["language_model"]),
                   device=get_available_device())
    optimizers = model.prepare_optimizers()

    for epoch in trange(config['num_epochs'], desc="Training"):
        run_epoch(dataloader=train_dataloader,
                  model=model,
                  optimizers=optimizers,
                  kg=kg_dataset,
                  config=config,
                  epoch=epoch)

        if epoch % config['save_each'] == 0 and epoch > 0:
            checkpoint_file_path = os.path.join(config['checkpoint_save_path'],
                                                f'sfavel_step_{epoch}.pt')

            logging.info(f"Saving model checkpoint at epoch {epoch} to {checkpoint_file_path}")
            torch.save(model, checkpoint_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training.json",
                        required=True,
                        help='Path to the json config for training')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    training_params = load_params_from_file(args.config_path)
    logging.info(f"Parameters for training: {training_params}")

    wandb.init(project="sfavel_pretraining", config=training_params)
    run_training(training_params)
