import os
import time
import json
import logging
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

def get_ram_memory_info():
    """Returns system RAM usage description."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        used = mem.used / (1024**3)
        total = mem.total / (1024**3)
        return f"RAM: {used:.1f}/{total:.1f}GB"
    except Exception:
        return None

def get_gpu_memory_info():
    """Returns GPU VRAM usage description."""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            return f"VRAM: {allocated:.1f}/{max_allocated:.1f}GB"
    except Exception:
        pass
    return None

def get_tpu_memory_info():
    """Returns TPU HBM memory usage description."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        info = xm.get_memory_info(device)
        free_gb = info.get("kb_free", 0) / (1024 * 1024)
        total_gb = info.get("kb_total", 0) / (1024 * 1024)
        used_gb = total_gb - free_gb
        return f"HBM: {used_gb:.1f}/{total_gb:.1f}GB"
    except Exception:
        return None

class ObservabilityCallback(TrainerCallback):
    """
    Hugging Face Trainer Callback to log system stats and metrics 
    to console and save them to a file for external tailing.
    """
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "training_metrics.jsonl")
        self.start_time = time.time()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        # Enrich logs dictionary with telemetry stats
        logs["step"] = state.global_step
        logs["max_steps"] = state.max_steps
        logs["elapsed_seconds"] = int(time.time() - self.start_time)
        
        ram_info = get_ram_memory_info()
        tpu_info = get_tpu_memory_info()
        gpu_info = get_gpu_memory_info()
        
        if ram_info:
            logs["ram"] = ram_info
        if tpu_info:
            logs["tpu_hbm"] = tpu_info
        elif gpu_info:
            logs["gpu_vram"] = gpu_info
            
        # Build clean console line
        console_parts = [f"Step {state.global_step}/{state.max_steps}"]
        if "loss" in logs:
            console_parts.append(f"Loss: {logs['loss']:.4f}")
        if "learning_rate" in logs:
            console_parts.append(f"LR: {logs['learning_rate']:.2e}")
        if ram_info:
            console_parts.append(ram_info)
        if tpu_info:
            console_parts.append(tpu_info)
        elif gpu_info:
            console_parts.append(gpu_info)
            
        # Log to stdout clearly
        print(f"Telemetry | {' | '.join(console_parts)}")
        
        # Save metrics to JSONL file
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(logs) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write metrics to JSONL: {e}")
