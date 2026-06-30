import logging
import numpy as np
import torch
import librosa
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    AutoFeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)
from src.decoding.ctc_decoder import create_ctc_decoder, decode_logits
from src.data.dataset import normalize_text

logger = logging.getLogger(__name__)

class VADSegmenter:
    """
    Splits long audio streams into non-silent segments to avoid GPU memory overflow
    and maintain CTC alignment accuracy.
    """
    def __init__(self, top_db=30, frame_length=2048, hop_length=512):
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length

    def segment(self, y, sr, min_duration=1.5, max_duration=30.0):
        # Use librosa split for robust energy-based silence detection
        intervals = librosa.effects.split(
            y, 
            top_db=self.top_db, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        )
        
        chunks = []
        for start, end in intervals:
            chunk = y[start:end]
            duration = len(chunk) / sr
            
            # Filter out micro-segments (noise fragments)
            if duration < min_duration:
                continue
                
            # If segment is longer than max_duration, chunk it linearly
            if duration > max_duration:
                max_samples = int(max_duration * sr)
                for step in range(0, len(chunk), max_samples):
                    sub_chunk = chunk[step : step + max_samples]
                    if len(sub_chunk) / sr >= min_duration:
                        chunks.append(sub_chunk)
            else:
                chunks.append(chunk)
                
        # If no intervals detected (e.g. uniform background hum), return the whole file
        if not chunks:
            return [y]
            
        return chunks

class LanguageIdentifier:
    """
    Zero-shot Language Identification classifier using MMS-LID,
    constrained specifically to the target classes (lin, sna, lug) via softmax masking.
    """
    def __init__(self, model_id="facebook/mms-lid-126", target_languages=["lin", "sna", "lug"]):
        logger.info(f"Loading Language Identification model: {model_id}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.target_languages = target_languages
        
        # Determine mapping from target languages to model class labels
        id2label = self.model.config.id2label
        self.target_indices = {}
        for idx, label in id2label.items():
            for lang in target_languages:
                if label.lower().startswith(lang):
                    self.target_indices[lang] = idx
                    
        logger.info(f"Language target indices mapping: {self.target_indices}")
        
    def identify(self, y, sr=16000):
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            
        inputs = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            
        # Perform target-restricted softmax/argmax
        target_scores = {}
        for lang, idx in self.target_indices.items():
            target_scores[lang] = logits[idx]
            
        # Argmax over target classes
        best_lang = max(target_scores, key=target_scores.get)
        logger.info(f"Identified language: {best_lang} (scores: {target_scores})")
        return best_lang

class ProductionASRPipeline:
    """
    Complete routing pipeline for Phase 2 zero-shot inference:
    Audio Stream -> VAD Chunker -> Language ID -> Adapter Swap -> CTC Beam Search -> Final Text
    """
    def __init__(
        self,
        base_model_id="facebook/mms-300m",
        target_languages=["lin", "sna", "lug"],
        adapter_paths=None, # dict mapping lang to checkpoint paths
        kenlm_paths=None, # dict mapping lang to KenLM .bin paths
        beam_width=128
    ):
        self.target_languages = target_languages
        self.beam_width = beam_width
        self.kenlm_paths = kenlm_paths or {}
        
        # Load base model and processors
        logger.info(f"Initializing base model: {base_model_id}")
        self.model = Wav2Vec2ForCTC.from_pretrained(base_model_id)
        self.processors = {}
        for lang in target_languages:
            # Each language has a tokenizer configuration
            proc = Wav2Vec2Processor.from_pretrained(base_model_id)
            proc.tokenizer.set_target_lang(lang)
            self.processors[lang] = proc
            
        # Load external adapters if checkpoints exist
        if adapter_paths:
            for lang, path in adapter_paths.items():
                logger.info(f"Loading customized adapter for {lang} from {path}")
                self.model.load_adapter(path)
                
        # Initialize VAD and LID
        self.vad = VADSegmenter()
        self.lid = LanguageIdentifier(target_languages=target_languages)
        
        # Build CTC decoders with KenLMs
        self.decoders = {}
        for lang in target_languages:
            lm_path = self.kenlm_paths.get(lang)
            vocab = self.processors[lang].tokenizer.get_vocab()
            self.decoders[lang] = create_ctc_decoder(
                vocab_dict=vocab,
                kenlm_model_path=lm_path,
                alpha=0.5, # tuned defaults
                beta=1.5
            )

    def transcribe(self, y, sr=16000):
        # 1. Voice Activity Detection Segmenting
        chunks = self.vad.segment(y, sr)
        
        # 2. Language Identification (run on full audio or longest chunk)
        longest_chunk = max(chunks, key=len)
        detected_lang = self.lid.identify(longest_chunk, sr)
        
        # 3. Route to designated adapter
        self.model.set_adapter(detected_lang)
        processor = self.processors[detected_lang]
        decoder = self.decoders[detected_lang]
        
        # Transcribe each segment
        transcriptions = []
        for idx, chunk in enumerate(chunks):
            # Resample segment
            if sr != 16000:
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                
            # Acoustic model forward pass
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits[0].cpu().numpy()
                
            # 4. Beam Search Decoding
            chunk_text = decode_logits(decoder, logits, beam_width=self.beam_width)
            normalized_chunk = normalize_text(chunk_text)
            if normalized_chunk:
                transcriptions.append(normalized_chunk)
                
        # Combine segments into a final sequence
        final_transcription = " ".join(transcriptions)
        return final_transcription, detected_lang
