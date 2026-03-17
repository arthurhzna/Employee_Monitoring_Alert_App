import ctypes
import pathlib
import sys
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from typing import Union
import cv2
from inference.base_model import BaseModel


class InsightFaceModel(BaseModel):
    def __init__(self, model_name: str,
        device: str,
        threshold: float = 0.55
    ) -> None:
        super().__init__(model_name, device)
        self.threshold = threshold

    def load(self) -> None:
        self._model = FaceAnalysis(name="buffalo_l")
        self._model.prepare(ctx_id=0)

        if self._device == "cuda":
            self._preload_cuda_runtime()
        
        self._is_loaded = True

    def _preload_cuda_runtime(self) -> None:
        site_pkgs = pathlib.Path(sys.prefix) / "lib/python3.10/site-packages/nvidia"
        lib_dirs = [
            site_pkgs / "cublas/lib",
            site_pkgs / "cudnn/lib",
            site_pkgs / "cuda_runtime/lib",
            site_pkgs / "cufft/lib",
        ]

        names = [
            "libcublasLt.so.11",
            "libcublas.so.11",
            "libcudnn.so.8",
            "libcudart.so.11.0",
            "libcufft.so.10",
        ]

        for lib_dir in lib_dirs:
            for name in names:
                candidate = lib_dir / name
                if candidate.exists():
                    try:
                        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    def predict(self, image: Union[str, np.ndarray],
        inputType: str = "frame") -> list:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        if inputType == "file":
            if not isinstance(image, str):
                raise TypeError(f"Expected str when inputType='file', got {type(image)}")
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif inputType == "frame":
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected np.ndarray when inputType='frame', got {type(image)}")
            image = image.copy() 
        else:
            raise ValueError(f"Invalid inputType: {inputType}. Must be 'file' or 'frame'")
        
        faces = self._model.get(image)
        return faces

    # @staticmethod
    # def cosine_similarity(a, b):
    #     """Calculate cosine similarity between two vectors."""
    #     return np.dot(a, b) / (norm(a) * norm(b))

    # def compare(self, probe_embedding, gallery_embeddings, gallery_labels):
    #     """
    #     Compare probe embedding dengan gallery embeddings.
        
    #     Args:
    #         probe_embedding: Embedding dari wajah yang dicari
    #         gallery_embeddings: Array of embeddings dari gallery
    #         gallery_labels: Array of labels dari gallery
            
    #     Returns:
    #         Tuple (name, similarity_score)
    #     """
    #     similarities = [
    #         self.cosine_similarity(probe_embedding, g)
    #         for g in gallery_embeddings
    #     ]

    #     best_idx = np.argmax(similarities)
    #     best_score = similarities[best_idx]
    #     second_best = sorted(similarities, reverse=True)[1]

    #     if best_score > self.threshold and (best_score - second_best) > 0.05:
    #         name = gallery_labels[best_idx]
    #     else:
    #         name = "UNKNOWN"

    #     return (name, best_score)

    def release(self):
        if self._model is not None:
            del self._model