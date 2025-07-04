from pathlib import Path

import librosa
import torch
from huggingface_hub import hf_hub_download
import os
import logging

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen

# Attempt to import comfyUI folder_paths to reuse its model directory logic
try:
    from folder_paths import base_path as _COMFY_BASE_PATH
except Exception:
    _COMFY_BASE_PATH = None

_logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"

# Ensure HF cache path is in shared models directory
if _COMFY_BASE_PATH is not None:
    _hf_cache_default = Path(_COMFY_BASE_PATH) / "public_models" / "hf_cache"
    os.environ.setdefault("HF_HOME", str(_hf_cache_default))

_REQUIRED_FILES_VC = ["s3gen.pt", "conds.pt"]


def _resolve_local_ckpt_dir_vc() -> Path | None:
    """Locate VC checkpoint directory using same strategy as TTS."""
    candidates = []

    env_dir = os.getenv("CHATTERBOX_CKPT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    if _COMFY_BASE_PATH is not None:
        _base = Path(_COMFY_BASE_PATH)
        candidates.extend([
            _base / "models" / "chatterbox",
            _base / "public_models" / "chatterbox",
            _base / "private_models" / "chatterbox",
        ])

    candidates.append(Path(__file__).parent / "assets" / "chatterbox")

    for cand in candidates:
        if cand.is_dir() and all((cand / f).is_file() for f in _REQUIRED_FILES_VC):
            _logger.info(f"ChatterboxVC: using local checkpoints at {cand}")
            return cand

    return None


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt")
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        ckpt_dir = _resolve_local_ckpt_dir_vc()

        if ckpt_dir is None:
            _logger.info("ChatterboxVC: local checkpoints not found, downloading from HF...")
            for fpath in _REQUIRED_FILES_VC:
                local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
            ckpt_dir = Path(local_path).parent

        return cls.from_local(ckpt_dir, device)

    def set_target_voice(self, target_voice):
        audio_wav = target_voice['waveform'].squeeze().numpy()
        audio_sr = target_voice['sample_rate']

        s3gen_ref_wav, _sr = librosa.resample(audio_wav, orig_sr=audio_sr, target_sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice=None,
    ):
        if target_voice:
            self.set_target_voice(target_voice)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice`"

        with torch.inference_mode():
            audio_wav = audio['waveform'].squeeze().numpy()
            audio_sr = audio['sample_rate']

            audio_16, _ = librosa.resample(audio_wav, orig_sr=audio_sr, target_sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.detach().cpu().unsqueeze(0)
        return {"waveform": wav, "sample_rate": S3GEN_SR}
