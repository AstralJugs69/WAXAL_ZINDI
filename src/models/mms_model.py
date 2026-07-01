import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoConfig

logger = logging.getLogger(__name__)

def load_processor_for_mms(model_id: str = "facebook/mms-300m", target_lang: str = "lin"):
    """
    Loads the Wav2Vec2 processor from mms-1b-all (which has per-language vocabulary
    configs) and sets the target language on the tokenizer.
    Note: mms-300m is a pre-trained backbone only — it has no vocabulary or adapter files.
          We load the processor from mms-1b-all which shares the same feature extraction
          pipeline but includes the language-specific vocabulary files.
    """
    processor_model_id = "facebook/mms-1b-all" if "mms-300m" in model_id else model_id
    logger.info(f"Loading Wav2Vec2 processor from {processor_model_id} with language {target_lang}")
    processor = Wav2Vec2Processor.from_pretrained(processor_model_id, target_lang=target_lang)
    processor.tokenizer.set_target_lang(target_lang)
    return processor


def get_mms_model_with_adapter(
    model_id: str = "facebook/mms-300m",
    target_lang: str = "lin",
    freeze_feature_extractor: bool = True,
    processor: Wav2Vec2Processor = None,
    torch_dtype = None
):
    """
    Loads facebook/mms-300m for CTC fine-tuning on a target language.

    Since mms-300m is a pre-trained backbone without language adapter files,
    we load it WITHOUT target_lang and then resize the CTC head (lm_head) to
    match the vocabulary size of our target language from the processor.
    This is the standard workflow for fine-tuning mms-300m from scratch on a
    new/low-resource language.
    """
    logger.info(f"Loading MMS model {model_id} as a CTC backbone for language {target_lang}")

    # Determine the vocab size from the processor (loaded from mms-1b-all)
    vocab_size = len(processor.tokenizer) if processor is not None else None

    # Load model WITHOUT target_lang — mms-300m has no adapter files
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id,
        ignore_mismatched_sizes=True,
        vocab_size=vocab_size,  # Resize LM head to match target language vocab
        torch_dtype=torch_dtype
    )

    # Freeze Wav2Vec2 feature encoder for CTC stability in low-resource setups
    if freeze_feature_extractor:
        logger.info("Freezing the Wav2Vec2 feature encoder layers.")
        model.freeze_feature_encoder()

    return model
