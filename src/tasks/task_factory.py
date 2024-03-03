from src.tasks.finetuning_fever_task import FineTuningFeverTask
from src.tasks.finetuning_fb15k237_task import FineTuningFB15K237Task

VALID_TASKS = ["FEVER", "FB15K-237"]


def get_finetuning_task(task_name):
    if task_name == "FEVER":
        return FineTuningFeverTask()
    elif task_name == "FB15K-237":
        return FineTuningFB15K237Task()
    else:
        raise ValueError(f"Unrecognized task name {task_name}. Valid: {VALID_TASKS}")
