"""Модуль для общих команд управления проектом."""

import fire

from llm_classification_finetuning.data import download_data

if __name__ == '__main__':
    fire.Fire(
        component={
            'download_data': download_data,
        }
    )
