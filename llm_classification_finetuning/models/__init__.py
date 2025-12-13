"""Пакет моделей: экспортирует доступные архитектуры и Lightning-модуль."""

from .baseline import LogisticRegression
from .deep_mlp import DeepMLP
from .module import LCFModelModule

__all__ = ['DeepMLP', 'LCFModelModule', 'LogisticRegression']
