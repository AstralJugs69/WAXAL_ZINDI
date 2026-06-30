import logging

logger = logging.getLogger(__name__)

def compute_audio_duration(audio_array, sampling_rate):
    """
    Computes duration of audio waveform in seconds.
    """
    if audio_array is None or len(audio_array) == 0:
        return 0.0
    return len(audio_array) / sampling_rate

def compute_speaking_rate(transcript, duration):
    """
    Computes speaking rate in words per second.
    """
    if not transcript or duration <= 0:
        return 0.0
    word_count = len(transcript.split())
    return word_count / duration

def filter_dataset(dataset_df, duration_min=1.5, duration_max=30.0, wps_min=1.0, wps_max=8.0):
    """
    Filters the dataset DataFrame using speech duration and speaking rate heuristics.
    Assumes dataset_df contains 'audio' (with 'array' and 'sampling_rate') and 'normalized_transcription'.
    """
    logger.info(f"Filtering dataset with constraints: duration [{duration_min}s, {duration_max}s], WPS [{wps_min}, {wps_max}]")
    initial_count = len(dataset_df)
    
    filtered_rows = []
    for idx, row in dataset_df.iterrows():
        audio_info = row.get("audio")
        transcript = row.get("normalized_transcription", "")
        
        # Safe checks for audio availability
        if not audio_info or "array" not in audio_info or "sampling_rate" not in audio_info:
            continue
            
        array = audio_info["array"]
        sr = audio_info["sampling_rate"]
        
        # 1. Compute duration and filter
        duration = compute_audio_duration(array, sr)
        if duration < duration_min or duration > duration_max:
            continue
            
        # 2. Compute speaking rate and filter
        speaking_rate = compute_speaking_rate(transcript, duration)
        if speaking_rate < wps_min or speaking_rate > wps_max:
            continue
            
        filtered_rows.append(row)
        
    filtered_df = dataset_df.from_records(filtered_rows) if filtered_rows else dataset_df.iloc[0:0]
    final_count = len(filtered_df)
    logger.info(f"Filtered dataset from {initial_count} to {final_count} samples (pruned {initial_count - final_count} rows)")
    return filtered_df
