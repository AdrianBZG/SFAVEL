from abc import ABC, abstractmethod


class FineTuningTask(ABC):
    def __init__(self):
        self.task_name = None

    @abstractmethod
    def run_train_step(self, args: dict):
        raise NotImplementedError("FineTuningTask subclasses must implement run_train_step method")

    @abstractmethod
    def run_eval_step(self, args: dict):
        raise NotImplementedError("FineTuningTask subclasses must implement run_eval_step method")
