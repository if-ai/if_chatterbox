import torch
import random
import numpy as np
import logging

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from comfy.utils import ProgressBar

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    

class Chatterbox_TTS:
    CATEGORY = "example"
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "optional": {
                "reference_voice": ("AUDIO",), 
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                },
            "required": {
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.01, "round": 0.01, "display": "number", "tooltip": "Exaggeration (Neutral = 0.5, extreme values can be unstable)"}),
                "cfg": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step":0.01, "round": 0.01, "tooltip": "CFG/Pace"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "control_after_generate": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step":0.01, "round": 0.01}),
                "device": (["cuda", "cpu"],),
                "text": ("STRING", {"multiline": True, "default": "What does the fox say?", "tooltip": "Text to synthesize"}),
                }
            }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "chatterbot_tts_process"
    CATEGORY = "ImpactFrames💥🎞️/audio"

    # cache
    _tts_model = None
    _tts_device = None

    def chatterbot_tts_process(self, text, exaggeration, cfg, seed, temperature, device, reference_voice=None, keep_model_loaded=True):

        # Check cached model
        if Chatterbox_TTS._tts_model is None or Chatterbox_TTS._tts_device != device:
            logger.info("Loading TTS model on %s", device)
            Chatterbox_TTS._tts_model = ChatterboxTTS.from_pretrained(device)
            Chatterbox_TTS._tts_device = device
        else:
            logger.info("Reusing cached TTS model on %s", device)

        model = Chatterbox_TTS._tts_model
        set_seed(int(seed))

        # -----------------------------------------------------------------------------
        # NEW: handle long texts by generating audio per chunk
        # -----------------------------------------------------------------------------

        text_chunks = chunk_text(text)

        logger.info(f"TTS: processing {len(text_chunks)} chunk(s)")

        wav_segments = []
        sample_rate = None
        message_lines = [f"Device: {device}", f"Total chunks: {len(text_chunks)}"]

        pbar = ProgressBar(100)

        for idx, chunk in enumerate(text_chunks, start=1):
            logger.info(f"Generating chunk {idx}/{len(text_chunks)} (words={len(chunk.split())})")

            # update progress roughly
            chunk_progress = int((idx-1)/len(text_chunks)*80)
            pbar.update_absolute(10 + chunk_progress)

            segment_dict = model.generate(
                chunk,
                audio_prompt=reference_voice,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg,
            )

            wav_tensor = segment_dict["waveform"]

            # Validate audio segment before adding
            is_ok, dur, err = validate_audio_segment(wav_tensor, idx, segment_dict.get("sample_rate", 16000))
            if is_ok:
                if sample_rate is None:
                    sample_rate = segment_dict.get("sample_rate", 16000)
                wav_segments.append(wav_tensor)
                logger.info("Generating chunk %s OK (%.2fs)", idx, dur)
                message_lines.append(f"Chunk {idx}/{len(text_chunks)} ✓ {dur:.2f}s")
            else:
                logger.warning("Skipping chunk %s: %s", idx, err)
                message_lines.append(f"Chunk {idx}/{len(text_chunks)} ✗ {err}")

        pbar.update_absolute(95)

        # Concatenate along the time dimension
        if len(wav_segments) == 1:
            wav = wav_segments[0]
        else:
            try:
                # Assume (batch?, channels, samples) – we want to concatenate on last dim
                wav = torch.cat(wav_segments, dim=-1)
            except RuntimeError:
                # Fallback concat on first dim if shape mismatch
                wav = torch.cat(wav_segments, dim=0)

        # Return AUDIO dictionary expected by ComfyUI
        audio_out = {"waveform": wav, "sample_rate": sample_rate or 16000}
        message = "\n".join(message_lines)

        # Optionally unload model to free VRAM
        if not keep_model_loaded:
            Chatterbox_TTS._tts_model = None
            Chatterbox_TTS._tts_device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.update_absolute(100)
        return (audio_out, message)


class Chatterbox_VC:
    CATEGORY = "example"
    @classmethod    
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_voice": ("AUDIO",), 
                "target_voice": ("AUDIO",), 
                "device": (["cuda", "cpu"],),
                },
            }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "chatterbot_vc_process"
    CATEGORY = "ImpactFrames💥🎞️/audio"

    _vc_model = None
    _vc_device = None

    def chatterbot_vc_process(self, reference_voice, target_voice, device, keep_model_loaded=True):

        # Model caching
        if Chatterbox_VC._vc_model is None or Chatterbox_VC._vc_device != device:
            logger.info("Loading VC model on %s", device)
            Chatterbox_VC._vc_model = ChatterboxVC.from_pretrained(device)
            Chatterbox_VC._vc_device = device
        else:
            logger.info("Reusing cached VC model on %s", device)

        model = Chatterbox_VC._vc_model

        message_lines = [f"Device: {device}"]

        try:
            wav = model.generate(
                audio=reference_voice,
                target_voice=target_voice,
            )
        except RuntimeError as e:
            if device == "cuda" and "CUDA" in str(e):
                logger.warning("CUDA error in VC; retrying on CPU")
                message_lines.append("CUDA error – retrying on CPU")
                # Clear cache and reload
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model_cpu = ChatterboxVC.from_pretrained(device="cpu")
                wav = model_cpu.generate(audio=reference_voice, target_voice=target_voice)
                if not keep_model_loaded:
                    model_cpu = None
            else:
                raise

        # Optionally unload model
        if not keep_model_loaded:
            Chatterbox_VC._vc_model = None
            Chatterbox_VC._vc_device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        audio_out = {"waveform": wav["waveform"], "sample_rate": wav["sample_rate"]}
        message = "\n".join(message_lines)
        return (audio_out, message)


NODE_CLASS_MAPPINGS = {
    "Chatterbox_TTS" : Chatterbox_TTS,
    "Chatterbox_VC" : Chatterbox_VC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Chatterbox_TTS": "Chatterbox TTS",
    "Chatterbox_VC": "Chatterbox VC",
}

# -----------------------------------------------------------------------------
# Utility: split long texts into sentence-aware chunks
# -----------------------------------------------------------------------------

def chunk_text(text: str, min_words: int = 80, max_words: int = 110):
    """Return a list of text chunks suitable for the TTS model.

    The algorithm keeps each chunk between *min_words* and *max_words* words and
    prefers to break at sentence boundaries (".", "!", "?").  If *text* is
    shorter than *max_words* words, a single-element list containing *text* is
    returned.
    """

    if not text or not text.strip():
        return ["You need to add some text for me to talk."]

    words = text.split()
    total_words = len(words)

    if total_words <= max_words:
        return [text]

    chunks = []
    current_position = 0

    while current_position < total_words:
        chunk_start = current_position
        min_end = min(chunk_start + min_words, total_words)
        max_end = min(chunk_start + max_words, total_words)

        # If we're near the end take the rest of the text.
        if max_end >= total_words:
            chunks.append(" ".join(words[chunk_start:]))
            break

        # Walk backwards from max_end looking for a punctuation boundary.
        split_position = None
        for pos in range(max_end - 1, min_end - 1, -1):
            if words[pos].endswith((".", "!", "?")):
                split_position = pos + 1  # include punctuation word
                break

        # Fallback: hard split at max_end
        if split_position is None:
            split_position = max_end

        chunks.append(" ".join(words[chunk_start:split_position]))
        current_position = split_position

    return chunks

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def validate_audio_segment(audio_tensor: torch.Tensor, chunk_index: int, sample_rate: int = 16000):
    """Return (is_valid, duration_seconds, error_msg or None)."""
    try:
        if audio_tensor is None or not isinstance(audio_tensor, torch.Tensor):
            return False, 0.0, f"Chunk {chunk_index}: not a valid tensor"

        if audio_tensor.numel() == 0:
            return False, 0.0, f"Chunk {chunk_index}: empty tensor"

        if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
            return False, 0.0, f"Chunk {chunk_index}: contains NaN/Inf"

        # Assume shape (channels, samples) or (batch, channels, samples)
        if audio_tensor.dim() == 2:
            num_samples = audio_tensor.shape[-1]
        elif audio_tensor.dim() == 3:
            num_samples = audio_tensor.shape[-1]
        else:
            return False, 0.0, f"Chunk {chunk_index}: unexpected shape {audio_tensor.shape}"

        duration = num_samples / sample_rate
        if duration < 0.1:
            return False, duration, f"Chunk {chunk_index}: too short ({duration:.2f}s)"
        if duration > 60.0:
            return False, duration, f"Chunk {chunk_index}: too long ({duration:.2f}s)"

        return True, duration, None

    except Exception as e:  # pylint: disable=broad-except
        return False, 0.0, f"Chunk {chunk_index}: validation error {e}"