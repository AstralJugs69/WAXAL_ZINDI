import logging
import numpy as np
import evaluate
from pyctcdecode import build_ctcdecoder
from src.data.dataset import normalize_text

logger = logging.getLogger(__name__)

def create_ctc_decoder(vocab_dict, kenlm_model_path=None, alpha=0.5, beta=1.5):
    """
    Creates a pyctcdecode CTC beam search decoder using vocabulary tokens and KenLM.
    """
    logger.info("Initializing pyctcdecode CTC decoder...")
    
    # Sort vocabulary dict by index to get token list
    sorted_vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]
    
    # Wav2Vec2 uses "|" as word boundary (space). We replace it for pyctcdecode if needed,
    # or pyctcdecode automatically maps standard symbols. Let's make sure it handles spaces.
    # We replace any tokenizer-specific space symbols like " " or "|" to matching characters.
    vocab_list = []
    for char in sorted_vocab:
        if char == "<pad>":
            vocab_list.append("") # pyctcdecode expects blank token or empty string
        elif char in ["|", " "]:
            vocab_list.append(" ")
        else:
            vocab_list.append(char)
            
    # Build decoding framework using KenLM
    decoder = build_ctcdecoder(
        labels=vocab_list,
        kenlm_model_path=kenlm_model_path,
        alpha=alpha,
        beta=beta
    )
    return decoder

def decode_logits(decoder, logits, beam_width=128, hotwords=None, hotword_weight=10.0):
    """
    Decodes frame-level acoustic logits using beam search with language model constraint.
    """
    if isinstance(logits, list):
        logits = np.array(logits)
        
    # Ensure logits are log-softmax or probabilities
    # pyctcdecode accepts unnormalized log probabilities (logits)
    text_prediction = decoder.decode(
        logits=logits,
        beam_width=beam_width,
        hotwords=hotwords,
        hotword_weight=hotword_weight
    )
    return text_prediction

def tune_decoder_hyperparameters(
    val_logits_list, 
    val_references, 
    vocab_dict, 
    kenlm_model_path, 
    n_trials=20
):
    """
    Runs a dynamic Optuna study to find the best alpha and beta values 
    for beam search decoding by minimizing validation error rate (0.5 * WER + 0.5 * CER).
    """
    logger.info("Starting hyperparameter tuning for pyctcdecode alpha and beta...")
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.0, 3.0)
        beta = trial.suggest_float("beta", 0.0, 5.0)
        
        # Build decoder with candidate parameters
        decoder = create_ctc_decoder(
            vocab_dict=vocab_dict,
            kenlm_model_path=kenlm_model_path,
            alpha=alpha,
            beta=beta
        )
        
        predictions = []
        for logits in val_logits_list:
            pred_text = decode_logits(decoder, logits, beam_width=64)
            predictions.append(normalize_text(pred_text))
            
        normalized_refs = [normalize_text(ref) for ref in val_references]
        
        # Filter empty references
        valid_preds = []
        valid_refs = []
        for p, r in zip(predictions, normalized_refs):
            if r.strip():
                valid_preds.append(p)
                valid_refs.append(r)
                
        if not valid_refs:
            return 1.0
            
        wer = wer_metric.compute(predictions=valid_preds, references=valid_refs)
        cer = cer_metric.compute(predictions=valid_preds, references=valid_refs)
        score = 0.5 * wer + 0.5 * cer
        return score
        
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"Optimal parameters found: alpha={best_params['alpha']:.4f}, beta={best_params['beta']:.4f}")
    logger.info(f"Best validation Score (0.5*WER + 0.5*CER): {best_score:.4f}")
    
    return best_params
