"""
Main finetuning entrypoint
"""

import argparse
import os
import sys
import logging
import torch
from torch import multiprocessing as torchmultiprocessing, nn
torchmultiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm, trange
import wandb
from dataset.data_handling import get_dataset, get_data_loader
from models.linear_classifier import Classifier
from utils import set_seed, load_params_from_file, get_lm_emb_dimensionality, get_available_device, load_model_from_checkpoint, GLOBAL_SEED, freeze_model
from tasks.task_factory import get_finetuning_task

logging.basicConfig(level=logging.INFO,
                    format='[finetuning:%(levelname)s] %(message)s')


def run_epoch(task, dataloader, sfavel, model, optimizer, kg, config, epoch=None):
    model.train()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        args = {"sfavel": sfavel, "model": model, "batch": batch, "kg": kg, "config": config}
        loss, log_metrics = task.run_train_step(args)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        wandb.log(log_metrics)
        logging.info(f'Epoch [{epoch+1}] Batch: {batch_idx} Metrics: {log_metrics}')


def run_evaluate(task, dataloader, sfavel, model, kg, config, epoch):
    model.eval()

    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        args = {"sfavel": sfavel, "model": model, "batch": batch, "kg": kg, "config": config}
        log_metrics = task.run_eval_step(args)

        wandb.log(log_metrics)
        logging.info(f'[DEV]: Epoch [{epoch+1}] Batch: {batch_idx} Metrics: {log_metrics}')

        total_loss += log_metrics["eval_loss"]

    return {"eval_loss": total_loss / len(dataloader)}


def prepare_datasets(config):
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

    dev_dataset, _ = get_dataset(root_path=config["root_path"],
                                 split="dev",
                                 fact_dataset=config["fact_dataset"],
                                 kg_dataset=config["kg_dataset"],
                                 node_feature_dim=get_lm_emb_dimensionality(config["language_model"]))

    dev_dataloader = get_data_loader(fact_dataset=dev_dataset,
                                     kg_dataset=kg_dataset,
                                     batch_size=config["batch_size"],
                                     shuffle=False,
                                     num_workers=0)

    return train_dataloader, kg_dataset, dev_dataloader


def run_finetuning(pretrained_model, config):
    train_dataloader, kg_dataset, dev_dataloader = prepare_datasets(config)
    best_model_metric = 0
    model = Classifier(config=config,
                       hidden_dim=config['classifier_hidden_dim'],
                       device=get_available_device())
    optimizer = model.prepare_optimizers()
    task = get_finetuning_task(config["task_name"])

    for epoch in trange(config['num_epochs'], desc="Training"):
        run_epoch(task=task,
                  dataloader=train_dataloader,
                  sfavel=pretrained_model,
                  model=model,
                  optimizer=optimizer,
                  epoch=epoch,
                  kg=kg_dataset,
                  config=config)

        if epoch % config['eval_each'] == 0:
            metrics = run_evaluate(task=task,
                                   dataloader=dev_dataloader,
                                   sfavel=pretrained_model,
                                   model=model,
                                   kg=kg_dataset,
                                   config=config,
                                   epoch=epoch)

            loss = metrics["eval_loss"]
            if loss < best_model_metric:
                logging.info(f"New BEST model (Loss: {loss}) at epoch {epoch} saving checkpoint. Task: {config['fact_dataset']}")
                checkpoint_file_path = os.path.join("model_checkpoints", f'{config["fact_dataset"]}_best_model_epoch_{epoch}.pt')
                torch.save(model, checkpoint_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/finetuning.json",
                        required=True,
                        help='Path to the json config for finetuning')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    finetuning_params = load_params_from_file(args.config_path)
    logging.info(f"Parameters for finetuning: {finetuning_params}")

    if finetuning_params["fact_dataset"] not in ["FEVER", "FB15K-237"]:
        raise ValueError(f"Unrecognized downstream task {finetuning_params['fact_dataset']}")

    finetuning_params.update({"task_name": finetuning_params["fact_dataset"]})

    model_path = os.path.join('model_checkpoints', f'sfavel_step_{finetuning_params["sfavel_checkpoint_step"]}.pt')
    pretrained_model = load_model_from_checkpoint(model_path)
    freeze_model(pretrained_model)  # Make sure pretrained backbone is frozen

    wandb.init(project="sfavel_finetuning", config=finetuning_params)
    run_finetuning(pretrained_model, finetuning_params)
