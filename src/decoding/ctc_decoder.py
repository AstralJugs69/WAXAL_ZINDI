import logging
import os
import numpy as np
import jiwer
from pyctcdecode import build_ctcdecoder
from src.data.dataset import normalize_text

logger = logging.getLogger(__name__)

def create_ctc_decoder(vocab_dict, kenlm_model_path=None, alpha=0.5, beta=1.5):
    """
    Creates a pyctcdecode CTC beam search decoder.

    Works with or without KenLM:
      - kenlm_model_path set → acoustic + n-gram LM (best WER)
      - kenlm_model_path None → pure beam search (still better than greedy argmax)

    Wav2Vec2 CTC blank is typically <pad>; word boundary is '|'.
    """
    lm_note = kenlm_model_path if kenlm_model_path else "no-LM (beam only)"
    logger.info(f"Initializing pyctcdecode CTC decoder ({lm_note})...")

    # Sort vocabulary by index so labels[i] matches logits[..., i]
    max_idx = max(vocab_dict.values())
    inv = [""] * (max_idx + 1)
    for tok, idx in vocab_dict.items():
        inv[idx] = tok

    vocab_list = []
    blank_set = False
    for tok in inv:
        if tok in ("<pad>",) or (tok == "" and not blank_set):
            # CTC blank → empty string (only one blank slot)
            vocab_list.append("")
            blank_set = True
        elif tok in ("|", " "):
            vocab_list.append(" ")
        elif tok in ("<s>", "</s>", "<unk>"):
            # non-emitting specials: use a rare placeholder char pyctcdecode will rarely pick
            vocab_list.append("\u2047")  # double question mark, stripped later if needed
        else:
            vocab_list.append(tok if tok is not None else "")

    kwargs = {"labels": vocab_list}
    if kenlm_model_path and os.path.exists(str(kenlm_model_path)):
        kwargs["kenlm_model_path"] = kenlm_model_path
        kwargs["alpha"] = alpha
        kwargs["beta"] = beta
    # Without KenLM, build_ctcdecoder still does useful beam search

    decoder = build_ctcdecoder(**kwargs)
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
            
        wer = jiwer.wer(reference=valid_refs, hypothesis=valid_preds)
        cer = jiwer.cer(reference=valid_refs, hypothesis=valid_preds)
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
