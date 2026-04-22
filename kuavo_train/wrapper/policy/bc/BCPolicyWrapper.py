from pathlib import Path
from typing import TypeVar
import builtins
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Unpack

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError

from lerobot.policies.pretrained import ActionSelectKwargs
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from kuavo_train.wrapper.policy.bc.BCConfigWrapper import CustomBCConfigWrapper
from kuavo_train.wrapper.policy.bc.BCModelWrapper import CustomBCModelWrapper


T = TypeVar("T", bound="CustomBCPolicyWrapper")


class CustomBCPolicyWrapper(PreTrainedPolicy):
    name = "bc"
    config_class = CustomBCConfigWrapper

    def __init__(self, config: CustomBCConfigWrapper):
        super().__init__(config)
        self.model = CustomBCModelWrapper(config)

    @property
    def config(self) -> CustomBCConfigWrapper:
        return self._config

    @config.setter
    def config(self, cfg: CustomBCConfigWrapper) -> None:
        self._config = cfg

    def reset(self):
        # Stateless BC policy: nothing to clear between episodes.
        return None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        del kwargs
        self.eval()
        return self.model(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Inference helper to keep interface aligned with other policies."""
        pred = self.predict_action_chunk(batch, **kwargs)
        if pred.dim() == 3:
            return pred[:, 0]  # first action step
        return pred

    def _compute_loss(self, pred: Tensor, target: Tensor, batch: dict[str, Tensor]) -> Tensor:
        if self.config.loss_type == "l2":
            base = F.mse_loss(pred, target, reduction="none")
        else:
            base = F.l1_loss(pred, target, reduction="none")

        if "action_is_pad" in batch and pred.dim() == 3 and batch["action_is_pad"].dim() == 2:
            # action_is_pad: (B, T)
            mask = (~batch["action_is_pad"]).unsqueeze(-1).to(base.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (base * mask).sum() / denom
        return base.mean()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        pred = self.model(batch)
        target = batch[ACTION]

        # Handle sequence/single-step mismatches conservatively.
        if pred.dim() == 2 and target.dim() == 3:
            target = target[:, 0, :]
        elif pred.dim() == 3 and target.dim() == 2:
            target = target.unsqueeze(1).expand(-1, pred.shape[1], -1)

        loss = self._compute_loss(pred, target, batch)
        loss_dict = {
            f"{self.config.loss_type}_loss": loss.item(),
        }
        return loss, loss_dict

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: CustomBCConfigWrapper | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        if config is None:
            config = CustomBCConfigWrapper.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    def get_optim_params(self):
        return [
            {
                "params": [p for p in self.parameters() if p.requires_grad],
                "lr": self.config.optimizer_lr,
            }
        ]
