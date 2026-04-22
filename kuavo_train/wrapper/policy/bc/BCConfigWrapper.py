from typing import Any, Dict
from dataclasses import dataclass, fields, field
import copy
from copy import deepcopy
from pathlib import Path
from typing import TypeVar

import draccus
from huggingface_hub.constants import CONFIG_NAME
from omegaconf import DictConfig, OmegaConf, ListConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig


T = TypeVar("T", bound="CustomBCConfigWrapper")


@PreTrainedConfig.register_subclass("custom_bc")
@dataclass
class CustomBCConfigWrapper(PreTrainedConfig):
    # Input/output features are injected by hydra instantiate in train script.
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # runtime / io
    device: str = "cpu"
    n_obs_steps: int = 1
    n_action_steps: int = 1

    # model
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1

    # training
    loss_type: str = "l1"  # l1 | l2
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1.0e-8
    optimizer_weight_decay: float = 1.0e-5

    normalization_mapping: dict[str, Any] = field(default_factory=dict)
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        default_map = {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }

        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged

        if isinstance(self.custom, (DictConfig, dict)):
            for k, v in self.custom.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(
                        f"Custom setting '{k}: {v}' conflicts with the parent base configuration. "
                        "Remove it from 'custom' and modify in the parent configuration instead."
                    )

        self._convert_omegaconf_fields()
        self.validate_features()

    def _convert_omegaconf_fields(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        return self.input_features.get("observation.state", None)

    @property
    def action_feature(self) -> PolicyFeature | None:
        return self.output_features.get("action", None)

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("BC requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("BC requires 'action' in output_features.")

        if self.loss_type not in {"l1", "l2"}:
            raise ValueError(f"Invalid loss_type='{self.loss_type}'. Supported: l1, l2.")

        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2 for a non-trivial MLP.")

    def get_optimizer_preset(self):
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        # Use the current step and previous steps according to n_obs_steps.
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int] | None:
        # Predict current action plus future actions according to n_action_steps.
        return list(range(self.n_action_steps))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        cfg_copy = deepcopy(self)
        if isinstance(cfg_copy.custom, dict):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        elif hasattr(cfg_copy, "custom") and hasattr(cfg_copy.custom, "keys"):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)

        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        parent_cls = PreTrainedConfig
        return parent_cls.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )
