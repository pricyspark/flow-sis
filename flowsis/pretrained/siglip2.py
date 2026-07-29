from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Literal
from collections.abc import Sequence
from transformers import SiglipTextConfig, SiglipTextModel, Siglip2Tokenizer

from .common import resolve_pretrained_source


class SigLIP2(nn.Module):
    def __init__(
        self,
        tokenizer,
        model,
        *,
        max_length: int = 64,
        return_mode: Literal["tokens", "pooled"] = "tokens",
        normalize: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        
        if return_mode not in ("tokens", "pooled"):
            raise ValueError(f"Unsupported return_mode: {return_mode}")
        
        self.max_length = int(max_length)
        self.return_mode = return_mode
        self.normalize = bool(normalize)
        
        self.tokenizer = tokenizer
        self.model = model
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        if device is not None:
            self.to(device)
            
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        *,
        cache_dir: str = "flowsis/models",
        max_length: int = 64,
        return_mode: Literal["tokens", "pooled"] = "tokens",
        normalize: bool = False,
        device: str | torch.device | None = None,
    ) -> SigLIP2:
        resolved_source, local_files_only = resolve_pretrained_source(
            model_name_or_path
        )
        
        # The SigLIP2 checkpoint's text_config still identifies itself as
        # `siglip_text_model`, so load the text tower with the matching class
        # to avoid a config/model-type mismatch warning in transformers.
        config = SiglipTextConfig.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        
        tokenizer = Siglip2Tokenizer.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        
        model = SiglipTextModel.from_pretrained(
            resolved_source,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
         
        return cls(
            tokenizer,
            model,
            max_length=max_length,
            return_mode=return_mode,
            normalize=normalize,
            device=device,
        )
    
    @torch.inference_mode()
    def forward(
        self,
        texts: str | Sequence[str],
    ) -> torch.Tensor:
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)
        
        device = self.device
        
        raw_tokens = self.tokenizer(
            batch,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )

        for i, input_ids in enumerate(raw_tokens["input_ids"]):
            if len(input_ids) > self.max_length:
                warnings.warn(
                    f"Input {i} has {len(input_ids)} tokens, "
                    f"which exceeds max_length={self.max_length}. "
                    f"It will be truncated by {len(input_ids) - self.max_length} tokens."
                )

        tokens = self.tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        
        outputs = self.model(**tokens)
        if self.return_mode == "tokens":
            embeddings = outputs.last_hidden_state  # (B,T,D)
        else:
            embeddings = outputs.pooler_output      # (B,D)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        if single_input:
            embeddings = embeddings[0]
            
        return embeddings
