import logging
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

def get_whisper_lora_model(
    model_id: str = "openai/whisper-small",
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list = ["q_proj", "v_proj"],
    lora_dropout: float = 0.05,
    load_in_8bit: bool = True,
    device: str = "cuda"
):
    """
    Loads Whisper-Small and wraps it with LoRA adapters for parameter-efficient fine-tuning.
    """
    logger.info(f"Loading Whisper model {model_id} with 8-bit={load_in_8bit}")
    
    # Configure quantization if loading in 8-bit
    kwargs = {}
    if load_in_8bit and torch.cuda.is_available():
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = device
        
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        **kwargs
    )
    
    # Prepare model for k-bit training
    if load_in_8bit:
        model = prepare_model_for_kbit_training(model)
        
    logger.info(f"Wrapping Whisper with LoRA: r={r}, alpha={lora_alpha}, targets={target_modules}")
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    # Force gradient checkpointing for memory optimization
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Verify trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_param} || "
        f"Trainable %: {100 * trainable_params / all_param:.4f}%"
    )
    
    return model
