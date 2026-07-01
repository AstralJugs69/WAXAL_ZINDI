import os
import random
import logging
import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def speed_perturb(y: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Stretches or compresses the raw audio waveform using linear interpolation (fast).
    """
    if speed_factor == 1.0 or len(y) == 0:
        return y
    indices = np.arange(0, len(y), speed_factor)
    indices = indices[indices < len(y)]
    return np.interp(indices, np.arange(len(y)), y)

def pitch_perturb(y: np.ndarray, sr: int, scale_factor: float) -> np.ndarray:
    """
    Modifies pitch of raw waveform. Uses librosa if available,
    otherwise falls back to speed-based pitch shifting.
    """
    if scale_factor == 1.0 or len(y) == 0:
        return y
    try:
        import librosa
        n_steps = 12.0 * np.log2(scale_factor)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except ImportError:
        # Fallback to simple speed perturbation which affects pitch
        return speed_perturb(y, 1.0 / scale_factor)

def time_domain_drop(y: np.ndarray, min_samples: int = 2000, max_samples: int = 3000) -> np.ndarray:
    """
    Replaces a random contiguous chunk of samples with zeros.
    """
    if len(y) <= max_samples:
        return y
    num_samples = random.randint(min_samples, max_samples)
    start_idx = random.randint(0, len(y) - num_samples)
    y_aug = y.copy()
    y_aug[start_idx : start_idx + num_samples] = 0.0
    return y_aug

def generate_pink_noise(length: int) -> np.ndarray:
    """
    Generates synthetic pink noise.
    """
    uneven = length % 2
    x = np.random.randn(length // 2 + 1 + uneven)
    s = np.arange(len(x)) + 1
    y = x / np.sqrt(s)
    z = np.fft.irfft(y, length + uneven)
    return z[:length]

def overlay_noise(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Overlays noise onto audio with a target Signal-to-Noise Ratio (SNR) in dB.
    """
    if len(y) == 0 or len(noise) == 0:
        return y
        
    # Standardize noise length
    if len(noise) < len(y):
        # Repeat noise if too short
        repeats = int(np.ceil(len(y) / len(noise)))
        noise = np.tile(noise, repeats)[:len(y)]
    else:
        # Slice noise if too long
        start_idx = random.randint(0, len(noise) - len(y))
        noise = noise[start_idx : start_idx + len(y)]
        
    p_signal = np.mean(y ** 2)
    p_noise = np.mean(noise ** 2)
    
    if p_signal == 0 or p_noise == 0:
        return y
        
    # Compute noise target power based on SNR
    p_noise_target = p_signal * (10 ** (-snr_db / 10.0))
    scale = np.sqrt(p_noise_target / (p_noise + 1e-10))
    
    return y + noise * scale

@dataclass
class DynamicAugmentator:
    noise_dir: str = None
    speed_range: tuple = (0.8, 1.2)
    pitch_range: tuple = (0.9, 1.1)
    snr_range: tuple = (5.0, 25.0)
    drop_range: tuple = (2000, 3000)
    
    def __post_init__(self):
        self.noise_files = []
        if self.noise_dir and os.path.exists(self.noise_dir):
            self.noise_files = [
                os.path.join(self.noise_dir, f) 
                for f in os.listdir(self.noise_dir) 
                if f.endswith(('.wav', '.mp3', '.flac'))
            ]
            logger.info(f"Loaded {len(self.noise_files)} noise files for augmentation.")

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        # 1. Speed perturbation (30% probability)
        if random.random() < 0.3:
            factor = random.uniform(*self.speed_range)
            y = speed_perturb(y, factor)
            
        # 2. Pitch modification (disabled due to librosa CPU overhead)
        # if random.random() < 0.3:
        #     factor = random.uniform(*self.pitch_range)
        #     y = pitch_perturb(y, sr, factor)
            
        # 3. Time drop-out chunks (30% probability)
        if random.random() < 0.3:
            y = time_domain_drop(y, *self.drop_range)
            
        # 4. Noise overlay (40% probability)
        if random.random() < 0.4:
            snr = random.uniform(*self.snr_range)
            # Pick a noise file or generate synthetic noise
            if self.noise_files and random.random() < 0.8:
                try:
                    noise_file = random.choice(self.noise_files)
                    noise, n_sr = sf.read(noise_file)
                    # Resample if sample rates mismatch
                    if n_sr != sr:
                        import librosa
                        noise = librosa.resample(noise, orig_sr=n_sr, target_sr=sr)
                    y = overlay_noise(y, noise, snr)
                except Exception as e:
                    logger.warning(f"Error loading noise file: {e}. Generating synthetic noise.")
                    noise = generate_pink_noise(len(y)) if random.random() < 0.5 else np.random.randn(len(y))
                    y = overlay_noise(y, noise, snr)
            else:
                noise = generate_pink_noise(len(y)) if random.random() < 0.5 else np.random.randn(len(y))
                y = overlay_noise(y, noise, snr)
                
        return y

class ASRDataCollatorWithPadding:
    """
    Data collator that dynamically pads inputs and targets for CTC (MMS) or Seq2Seq (Whisper).
    Applies waveform augmentations on-the-fly.
    """
    def __init__(
        self, 
        processor: Any, 
        augmentator: DynamicAugmentator = None,
        is_seq2seq: bool = False,
        sampling_rate: int = 16000,
        static_buckets: bool = False
    ):
        self.processor = processor
        self.augmentator = augmentator
        self.is_seq2seq = is_seq2seq
        self.sampling_rate = sampling_rate
        self.static_buckets = static_buckets

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        # Audio preprocessing
        input_features = []
        
        # Resolve target sequence length for static bucketing (for CTC inputs)
        target_audio_len = None
        if self.static_buckets and not self.is_seq2seq:
            max_len = max(len(feature["audio"]["array"]) for feature in features)
            # quantiles-based bucket sizes (at 16kHz)
            # representing 1.5s, 3.0s, 6.0s, 12.0s, 18.0s, 24.0s, 30.0s
            audio_buckets = [24000, 48000, 96000, 192000, 288000, 384000, 480000]
            target_audio_len = audio_buckets[-1]
            for b in audio_buckets:
                if max_len <= b:
                    target_audio_len = b
                    break

        for feature in features:
            audio_info = feature["audio"]
            y = audio_info["array"]
            sr = audio_info["sampling_rate"]
            
            # Ensure target sampling rate (usually 16000)
            if sr != self.sampling_rate:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sampling_rate)
                sr = self.sampling_rate
                
            # Apply dynamic augmentations only during training (when grad is enabled)
            if self.augmentator is not None and torch.is_grad_enabled():
                # Avoid pitch perturbation as librosa pitch shifting is extremely slow on CPU
                # We still keep speed, time drop, and noise overlay
                y = self.augmentator(y, sr)
                
            # Pad to the resolved static bucket size
            if target_audio_len is not None:
                if len(y) < target_audio_len:
                    y = np.pad(y, (0, target_audio_len - len(y)), 'constant')
                else:
                    y = y[:target_audio_len]
                
            # Process waveform
            if self.is_seq2seq:
                # Whisper features (always outputs exactly 3000 frames)
                processed = self.processor.feature_extractor(y, sampling_rate=sr, return_tensors="pt")
                input_features.append({"input_features": processed.input_features[0]})
            else:
                # MMS/CTC inputs
                processed = self.processor(y, sampling_rate=sr, return_tensors="pt")
                input_features.append({"input_values": processed.input_values[0]})
                
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, 
            return_tensors="pt"
        )
        
        # Tokenize labels
        label_features = []
        for feature in features:
            text = feature.get("normalized_transcription") or feature.get("transcription")
            if text:
                tokenized = self.processor.tokenizer(text)
                label_features.append({"input_ids": tokenized.input_ids})
                
        if label_features:
            # Resolve target label length for static bucketing
            if self.static_buckets:
                max_label_len = max(len(l["input_ids"]) for l in label_features)
                label_buckets = [16, 32, 64, 96, 128, 192, 256]
                target_label_len = label_buckets[-1]
                for b in label_buckets:
                    if max_label_len <= b:
                        target_label_len = b
                        break
                # Pad token IDs manually to static target label length
                pad_id = self.processor.tokenizer.pad_token_id or 0
                for l in label_features:
                    curr_len = len(l["input_ids"])
                    if curr_len < target_label_len:
                        l["input_ids"] = l["input_ids"] + [pad_id] * (target_label_len - curr_len)
                    else:
                        l["input_ids"] = l["input_ids"][:target_label_len]
                        
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                return_tensors="pt"
            )
            # Replace padding token id with -100 to ignore loss
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            
        return batch
