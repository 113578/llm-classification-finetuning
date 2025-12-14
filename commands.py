"""Модуль для общих команд управления проектом."""

import fire

from llm_classification_finetuning.data import download_data
from llm_classification_finetuning.infer import infer
from llm_classification_finetuning.train import train

if __name__ == '__main__':
    fire.Fire(component={'download_data': download_data, 'train': train, 'infer': infer})
