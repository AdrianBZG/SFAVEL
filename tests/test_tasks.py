import pytest
from src.tasks.task_factory import get_finetuning_task
from src.tasks.finetuning_task import FineTuningTask
from src.tasks.finetuning_fb15k237_task import FineTuningFB15K237Task
from src.tasks.finetuning_fever_task import FineTuningFeverTask

def test_get_finetuning_task():
    task = get_finetuning_task("FEVER")
    assert isinstance(task, FineTuningFeverTask)

    task = get_finetuning_task("FB15K-237")
    assert isinstance(task, FineTuningFB15K237Task)

    with pytest.raises(ValueError):
        get_finetuning_task("INVALID_TASK")

class DummyFineTuningTask(FineTuningTask):
    def run_train_step(self, args: dict):
        pass

    def run_eval_step(self, args: dict):
        pass

def test_finetuning_task():
    task = DummyFineTuningTask()
    assert task.task_name is None

    with pytest.raises(NotImplementedError):
        super(DummyFineTuningTask, task).run_train_step({})

    with pytest.raises(NotImplementedError):
        super(DummyFineTuningTask, task).run_eval_step({})

def test_finetuning_fb15k237_task():
    task = FineTuningFB15K237Task()
    assert task.task_name == "FB15K-237"

    with pytest.raises(NotImplementedError):
        task.run_train_step({})

    with pytest.raises(NotImplementedError):
        task.run_eval_step({})
