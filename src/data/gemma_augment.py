import logging
import functools
import numpy as np
import torch
from collections.abc import Mapping, Sequence
from typing import Any
import transformers

logger = logging.getLogger(__name__)

def _mask_labels(
    labels: torch.Tensor,
    proc: transformers.AutoProcessor,
) -> torch.Tensor:
    """Sets padding and special-token positions to -100 (ignored by CE loss)."""
    labels = labels.clone()
    tokenizer = proc.tokenizer
    special_attrs = [
        "pad_token_id",
        "image_token_id",
        "audio_token_id",
        "boi_token_id",
        "eoi_token_id",
    ]
    mask_ids = [
        getattr(tokenizer, attr)
        for attr in special_attrs
        if getattr(tokenizer, attr, None) is not None
    ]
    if mask_ids:
        labels[
            torch.isin(labels, torch.tensor(mask_ids, device=labels.device))
        ] = -100
    return labels


def collate_fn(
    examples: Sequence[Mapping[str, Any]],
    proc: transformers.AutoProcessor,
    audio_cache: dict = None,
    system_message: str = "You are an assistant that transcribes speech accurately.",
    user_message: str = "Please transcribe this audio.",
) -> dict[str, Any]:
    """Collates a list of examples into a training batch by dynamically applying chat templates."""
    texts = []
    audios = []
    
    for ex in examples:
        # Extract audio array, utilizing RAM cache if available
        audio_info = ex.get("audio")
        path = audio_info.get("path") if isinstance(audio_info, dict) else getattr(audio_info, "path", "")
        
        if audio_cache and path in audio_cache:
            y, sr = audio_cache[path]
            arr = np.asarray(y).flatten()
        elif isinstance(audio_info, dict) and "array" in audio_info:
            arr = np.asarray(audio_info["array"]).flatten()
        else:
            from src.data.dataset import get_audio_data
            y, sr = get_audio_data(audio_info)
            arr = np.asarray(y).flatten() if y is not None else np.zeros(16000)
            
        audios.append(arr)
        
        # Format the text prompt using the chat template dynamically
        transcription = ex.get("transcription") or ex.get("normalized_transcription") or ""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": arr},
                    {"type": "text", "text": user_message},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(transcription)}],
            },
        ]
        
        chat_prompt = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(chat_prompt)

    batch = proc(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    batch = {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch["labels"] = _mask_labels(batch["input_ids"], proc)
    return batch


class GemmaDataCollator:
    """
    Modular data collator for Google Gemma 3n/4n multimodal fine-tuning.
    Supports optional memory caching of audio waveforms.
    """
    def __init__(self, processor, audio_cache=None, system_message=None, user_message=None):
        self.processor = processor
        self.audio_cache = audio_cache if audio_cache is not None else {}
        self.system_message = system_message or "You are an assistant that transcribes speech accurately."
        self.user_message = user_message or "Please transcribe this audio."

    def __call__(self, examples):
        return collate_fn(
            examples, 
            self.processor, 
            self.audio_cache,
            system_message=self.system_message,
            user_message=self.user_message
        )
