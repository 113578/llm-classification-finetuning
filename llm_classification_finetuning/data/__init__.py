"""Пакет data: экспортирует LCFDataModule и утилиту для загрузки датасета."""

from .datamodule import LCFDataModule
from .download_data import download_data

__all__ = ['LCFDataModule', 'download_data']
