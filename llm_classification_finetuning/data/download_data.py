"""Утилита для скачивания датасета с Kaggle через kagglehub.

Функция download_data скачивает датасет по handle и помещает его в указанный каталог.
"""

import shutil

import kagglehub


def download_data(dataset_url: str, target_dir: str) -> None:
    """
    Скачивает датасет с kagglehub.dataset_download и копирует в target_dir.

    Parameters
    ----------
    dataset_url : str
        Handle датасета на Kaggle (например 'owner/dataset').
    target_dir : str
        Директория, куда будет скопирован датасет.

    Returns
    -------
    None
    """
    data_path = kagglehub.dataset_download(handle=dataset_url)

    shutil.copytree(src=data_path, dst=target_dir, dirs_exist_ok=True)
    shutil.rmtree(data_path)
