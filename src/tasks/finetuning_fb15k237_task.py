from src.tasks.finetuning_task import FineTuningTask


class FineTuningFB15K237Task(FineTuningTask):
    def __init__(self):
        super().__init__()
        self.task_name = "FB15K-237"

    def run_train_step(self, args: dict):
        raise NotImplementedError(f'Not implemented yet for {self.task_name} task.')

    def run_eval_step(self, args: dict):
        raise NotImplementedError(f'Not implemented yet for {self.task_name} task.')
