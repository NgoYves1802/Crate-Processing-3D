"""
crate_vision/ai_verifier.py
============================
DenseNet-121 crate classifier.

Supports two backends (tried in order):
  1. PyTorch  (.pth weights)  — requires torch + torchvision
  2. ONNX Runtime (.onnx)     — requires onnxruntime

Usage
-----
    from crate_vision.ai_verifier import CrateVerifier
    from crate_vision.config import get_config

    verifier = CrateVerifier.from_config(get_config())
    result   = verifier.verify(amp_crop_uint8)

    # result dict keys:
    #   passed     : bool | None
    #   label      : str  | None
    #   confidence : float | None
    #   threshold  : float
    #   model_used : "pytorch" | "onnx" | "skipped" | "error"
    #   error      : str | None

The module-level singleton is kept for pipeline convenience:

    from crate_vision.ai_verifier import get_verifier
    result = get_verifier().verify(crop)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import densenet121

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class CrateVerifier:
    """Thin wrapper around the DenseNet-121 crate classifier."""

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model=None,
        session=None,
        backend: str = "skipped",
        class_names: list[str] | None = None,
        crate_idx: int = 0,
        conf_threshold: float = 0.70,
    ):
        self._model = model
        self._session = session
        self.backend = backend
        self.class_names = class_names or ["crate", "no_crate"]
        self.crate_idx = crate_idx
        self.conf_threshold = conf_threshold

        self._tf = None
        if _TORCH_AVAILABLE:
            self._tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ])

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config) -> "CrateVerifier":
        """
        Build a CrateVerifier from a CrateVisionConfig.
        Tries PyTorch first, then ONNX, then returns a no-op verifier.
        """
        pth_path  = config.ai_model_pth
        onnx_path = config.ai_model_onnx
        threshold = config.ai_conf_threshold
        classes   = config.ai_class_names
        crate_idx = config.ai_crate_class_idx

        # Try PyTorch
        if pth_path and _TORCH_AVAILABLE:
            try:
                model = cls._load_pytorch_model(pth_path, len(classes))
                return cls(model=model, backend="pytorch",
                           class_names=classes, crate_idx=crate_idx,
                           conf_threshold=threshold)
            except Exception:
                pass

        # Try ONNX
        if onnx_path and _ONNX_AVAILABLE:
            try:
                session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                return cls(session=session, backend="onnx",
                           class_names=classes, crate_idx=crate_idx,
                           conf_threshold=threshold)
            except Exception:
                pass

        return cls(backend="skipped", class_names=classes,
                   crate_idx=crate_idx, conf_threshold=threshold)

    @staticmethod
    def _load_pytorch_model(pth_path: str, num_classes: int):
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        state_dict = torch.load(pth_path, map_location=device)
        
        # Check if this is a full model (with 'features.' keys) or just classifier
        has_full_model = any(k.startswith('features.') for k in state_dict.keys())
        
        if has_full_model:
            # Full model saved - load entire DenseNet-121
            model = densenet121(weights=None)
            # Replace classifier with our custom head
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes),
            )
            # Load only matching keys (features + our classifier)
            model.load_state_dict(state_dict, strict=False)
        else:
            # Only classifier saved - use new architecture and load weights
            model = densenet121(weights=None)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes),
            )
            model.load_state_dict(state_dict)
        
        model.eval()
        model.to(device)
        return model

    # ── Inference ─────────────────────────────────────────────────────────────

    def verify(self, amp_crop: np.ndarray) -> dict:
        """
        Run AI verification on an amplitude crop.

        Parameters
        ----------
        amp_crop : uint8 or float [0,1] grayscale/RGB array (any size)

        Returns
        -------
        dict: passed, label, confidence, threshold, model_used, error
        """
        base = {
            "passed":     None,
            "label":      None,
            "confidence": None,
            "threshold":  self.conf_threshold,
            "model_used": self.backend,
            "error":      None,
        }

        if self.backend == "skipped":
            return base

        try:
            pil_img = self._to_pil(amp_crop)

            if self.backend == "pytorch":
                result = self._infer_pytorch(pil_img)
            else:
                result = self._infer_onnx(pil_img)

            label      = result["label"]
            confidence = result["confidence"]
            passed     = (
                label == self.class_names[self.crate_idx]
                and confidence >= self.conf_threshold
            )
            base.update({
                "passed":     bool(passed),
                "label":      label,
                "confidence": round(confidence, 4),
            })

        except Exception as exc:
            base.update({"passed": False, "error": str(exc)})

        return base

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(arr: np.ndarray) -> Image.Image:
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
        return Image.fromarray(arr).convert("RGB")

    def _infer_pytorch(self, pil_img: Image.Image) -> dict:
        device = next(self._model.parameters()).device
        tensor = self._tf(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {"label": self.class_names[idx], "confidence": float(probs[idx])}

    def _infer_onnx(self, pil_img: Image.Image) -> dict:
        img  = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        mean = np.array(self._MEAN, dtype=np.float32)
        std  = np.array(self._STD,  dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)[np.newaxis, :]
        out  = self._session.run(None, {self._session.get_inputs()[0].name: img})[0][0]
        exp_s = np.exp(out - out.max())
        probs = exp_s / exp_s.sum()
        idx   = int(np.argmax(probs))
        return {"label": self.class_names[idx], "confidence": float(probs[idx])}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_VERIFIER: CrateVerifier | None = None


def get_verifier(config=None) -> CrateVerifier:
    """
    Return the module-level CrateVerifier singleton.

    If *config* is provided the first time, it initialises from that config.
    Subsequent calls ignore *config* and return the existing instance.
    """
    global _VERIFIER
    if _VERIFIER is None:
        if config is None:
            from crate_vision.config import get_config
            config = get_config()
        _VERIFIER = CrateVerifier.from_config(config)
    return _VERIFIER


def reset_verifier() -> None:
    """Discard the singleton — useful for testing or config hot-reload."""
    global _VERIFIER
    _VERIFIER = None
