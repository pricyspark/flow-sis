import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flowsis.utils import resolve_pretrained_source
from transformers import Siglip2TextModel, Siglip2TextConfig, Siglip2Tokenizer
from collections.abc import Sequence
from typing import Literal


class SigLIP2(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        *,
        cache_dir: str = "flowsis/models",
        max_length: int = 64,
        return_mode: Literal["tokens", "pooled"] = "tokens",
        normalize: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        
        if return_mode not in ("tokens", "pooled"):
            raise ValueError(f"Unsupported return_mode: {return_mode}")
        
        self.max_length = int(max_length)
        self.return_mode = return_mode
        self.normalize = bool(normalize)
        
        resolved_source, local_files_only = resolve_pretrained_source(
            model_name_or_path, 
            cache_dir,
        )
        
        config = Siglip2TextConfig.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        
        config.output_hidden_states = output_hidden_states
        config.output_attentions = output_attentions
        
        self.tokenizer = Siglip2Tokenizer.from_pretrained(
            resolved_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        
        self.model = Siglip2TextModel.from_pretrained(
            resolved_source,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        if device is not None:
            self.to(device)
            
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    
    @torch.inference_mode()
    def forward(
        self,
        texts: str | Sequence[str],
        *,
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, object]: # not sure object or Any
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)
        
        device = self.device
        
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
            
        if return_outputs:
            return embeddings, outputs
        
        return embeddings