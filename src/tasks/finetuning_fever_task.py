import torch
from src.tasks.finetuning_task import FineTuningTask


class FineTuningFeverTask(FineTuningTask):
    def __init__(self):
        super().__init__()
        self.task_name = "FEVER"

    def run_train_step(self, args: dict):
        # Unpack arguments
        sfavel, model, batch, kg, config = args.values()

        with torch.no_grad():
            x_lm, candidates_z = sfavel.inference(batch, kg, k=config["k"])

        y_pred = model(x_lm, candidates_z)
        y_true = torch.stack([claim["label"] for claim in batch])

        loss = model.calculate_loss(y_pred, y_true)
        accuracy = (y_pred.round() == y_true).float().mean()

        log_metrics = {"train_loss": loss.item(), "train_acc": accuracy.item()}
        return loss, log_metrics

    def run_eval_step(self, args: dict):
        # Unpack arguments
        sfavel, model, batch, kg, config = args.values()

        y_true = torch.stack([claim["label"] for claim in batch])

        with torch.no_grad():
            x_lm, candidates_z = sfavel.inference(batch, kg, k=config["k"])
            y_pred = model(x_lm, candidates_z)
            loss = model.calculate_loss(y_pred, y_true)
            accuracy = (y_pred.round() == y_true).float().mean()

        log_metrics = {"eval_loss": loss.item(), "eval_acc": accuracy.item()}
        return log_metrics
