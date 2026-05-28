"""Run-folder and metadata management for training results."""

import time
from datetime import datetime
from pathlib import Path

import torch

from . import utils

__all__ = ['RunManager']


class RunManager:
    """Create unique model directories and save run metadata.

    ``ModelHandler`` owns training behavior; ``RunManager`` only handles the
    filesystem convention for saved runs.
    """

    def __init__(
            self,
            save_path: str | Path = 'Model',
            model_name: str = None
    ):
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.original_model_name = model_name
        self.model_path = None

    @staticmethod
    def model_folder_timestamp() -> str:
        """Return a timestamp suitable for result directory names."""

        now = datetime.now()
        return f'{now:%y%m%d-%H%M%S}-{now.microsecond // 1000:03d}'

    def prepare_model_folder(self) -> None:
        """Create a unique result directory under ``save_path``."""

        base_model_name = self.original_model_name
        for _ in range(1000):
            timestamp = self.model_folder_timestamp()
            if base_model_name is None:
                self.model_name = timestamp
            else:
                self.model_name = f'{timestamp}_{base_model_name}'
            self.model_path = self.save_path / self.model_name
            try:
                self.model_path.mkdir(parents=True, exist_ok=False)
                return
            except FileExistsError:
                time.sleep(0.001)
        raise RuntimeError(f'Failed to create a unique model folder under {self.save_path}.')

    def save_metadata(self, model_handler) -> None:
        """Save configuration JSON files and lightweight data summaries."""

        config = {
            'format_version': 1,
            'nn': model_handler.nn.config_dict(),
            'optimizer': model_handler.optimizer.config_dict(),
            'model_handler': model_handler.config_dict(),
            'data_fitting': model_handler.data_fitting.config_dict() if model_handler.has_data else None,
            'data_handler': model_handler.data_fitting.data_handler.config_dict() if model_handler.has_data else None,
            'regularization': model_handler.regularization.config_dict() if model_handler.has_reg else None
        }
        utils.save_json(self.model_path / 'config.json', config)

        metadata_path = self.model_path / 'metadata'
        utils.save_json(metadata_path / 'nn.json', config['nn'])
        utils.save_json(metadata_path / 'optimizer.json', config['optimizer'])
        utils.save_json(metadata_path / 'model_handler.json', config['model_handler'])
        if model_handler.has_data:
            utils.save_json(metadata_path / 'data_fitting.json', config['data_fitting'])
            utils.save_json(metadata_path / 'data_handler.json', config['data_handler'])
            model_handler.data_fitting.data_handler.save_summary(metadata_path / 'data_summary.json')
            model_handler.data_fitting.data_handler.save_summary(metadata_path / 'data_summary.csv')
            if model_handler.save_model:
                torch.save(
                    model_handler.data_fitting.data_handler.normalizer_dict(),
                    self.model_path / 'normalizer.pth'
                )
        if model_handler.has_reg:
            utils.save_json(metadata_path / 'regularization.json', config['regularization'])
            for idx, input_generator_ in enumerate(model_handler.regularization.input_generators):
                utils.save_json(metadata_path / f'input_generator_{idx}.json', input_generator_.config_dict())
