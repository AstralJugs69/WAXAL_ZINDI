import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)

def get_mms_model_with_adapter(
    model_id: str = "facebook/mms-300m",
    target_lang: str = "lin",
    freeze_feature_extractor: bool = True
):
    """
    Loads facebook/mms-300m with a target language-specific adapter,
    optionally freezing the feature extractor for stable fine-tuning.
    """
    logger.info(f"Loading MMS model {model_id} for target language {target_lang}")
    
    # Load model and set target language adapter
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id,
        target_lang=target_lang,
        ignore_mismatched_sizes=True
    )
    
    # Freeze Wav2Vec2 feature encoder for CTC stability in low-resource setups
    if freeze_feature_extractor:
        logger.info("Freezing the Wav2Vec2 feature extractor layers.")
        model.freeze_feature_extractor()
        
    return model

def load_processor_for_mms(model_id: str = "facebook/mms-300m", target_lang: str = "lin"):
    """
    Loads the Wav2Vec2 processor and sets the target language on the tokenizer.
    """
    # Use "facebook/mms-1b-all" for processor configuration to get vocabulary files
    # since base "mms-300m" does not contain ASR vocabulary configs.
    processor_model_id = "facebook/mms-1b-all" if "mms-300m" in model_id else model_id
    logger.info(f"Loading Wav2Vec2 processor for {processor_model_id} with language {target_lang}")
    processor = Wav2Vec2Processor.from_pretrained(processor_model_id, target_lang=target_lang)
    processor.tokenizer.set_target_lang(target_lang)
    return processor
