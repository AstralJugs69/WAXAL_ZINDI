import logging
import torch
import transformers

logger = logging.getLogger(__name__)

def load_processor_for_gemma(model_id: str = "google/gemma-3n-E2B-it"):
    """
    Loads AutoProcessor for Gemma 3n/4n models, ensuring tokenizer configurations
    match the requirements for ASR fine-tuning.
    """
    logger.info(f"Loading Gemma processor from {model_id}...")
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    return processor


def get_gemma_model(
    model_id: str = "google/gemma-3n-E2B-it",
    torch_dtype = torch.bfloat16,
    device_map = "auto",
):
    """
    Loads Gemma3nForConditionalGeneration from pretrained.
    Important: timm must be imported BEFORE transformers loads the model,
    which is done in the main script/imports.
    """
    logger.info(f"Loading Gemma 3n model {model_id} with dtype {torch_dtype}...")
    model = transformers.Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return model
