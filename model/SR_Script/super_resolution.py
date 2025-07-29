import os
from typing import Any, Dict, List, Tuple, Optional  
import threading

import torch  # NOQA
import numpy as np
import onnxruntime as ort

from cryptography.fernet import Fernet

from .tiling_image_loader import SA_Tiling_ImageLoader


class SA_SuperResolution:
    """
    A class to apply super-resolution on images using a specific ONNX model.

    Supports thread-safe inference via internal lock.
    For multiprocessing, instantiate one object per process.
    """

    def __init__(
        self,
        models_dir: str,
        model_scale: int,
        tile_size: int = 128,
        gpu_id: int = 0,
        verbosity: bool = False,
    ) -> None:
        """
        Initialize with model directory, scale, and device info.

        Args:
            models_dir (str): Directory path to encrypted model files.
            model_scale (int): Super-resolution scale factor.
            tile_size (int): Tile size for image loader (default 128).
            gpu_id (int): GPU index (>=0 for GPU, -1 for CPU).
            verbosity (bool): Print debug info.
        """
        self.scale: int = model_scale
        self.tile_size: int = tile_size
        self.encrypted_model_path: str = self._model_definition(models_dir)

        # Thread lock to make ONNX Runtime calls thread-safe
        self.lock = threading.Lock()

        self.network, self.input_name = self._decrypt_model(gpu_id, verbosity)
        self.dataloader: SA_Tiling_ImageLoader = SA_Tiling_ImageLoader(self.tile_size)

    def _model_definition(self, models_dir: str) -> str:
        return os.path.join(models_dir, f"edsr_{self.scale}x.ven")

    def _decrypt_model(
        self,
        gpu_id: int,
        verbosity: bool = False,
        decryption_key: Optional[bytes] = None
    ) -> Tuple[ort.InferenceSession, str]:
        """
        Decrypt the ONNX model and initialize ONNX Runtime session.

        Returns:
            model (ort.InferenceSession): ONNX runtime model.
            input_name (str): Name of the input tensor.
        """
        with open(self.encrypted_model_path, "rb") as encrypted_file:
            encrypted_model = encrypted_file.read()

        if decryption_key is None:
            key = b"LtBDDJTE04l7Kef4PiYTa21RX4svq1vcGRbBkW_ZSwc="
        else:
            key = decryption_key

        fernet = Fernet(key)
        decrypted_model = fernet.decrypt(encrypted_model)

        # Default to CPU provider
        providers: List[Any] = ["CPUExecutionProvider"]

        # If GPU requested and available, add CUDA provider with config
        if gpu_id >= 0 and ort.get_device() == "GPU":
            providers.insert(0, (
                "CUDAExecutionProvider",
                {
                    "device_id": gpu_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
            ))

        try:
            model = ort.InferenceSession(decrypted_model, providers=providers)
            input_name = model.get_inputs()[0].name
            model_name = os.path.basename(self.encrypted_model_path)

            if verbosity:
                print(f"✅ ONNX providers available: {ort.get_available_providers()}")
                print(f"✅ ONNX session using: {model.get_providers()}")

                if "CUDAExecutionProvider" in model.get_providers():
                    print(f"🚀 {model_name} initialized on GPU (CUDA)")
                else:
                    print(f"🖥️ {model_name} initialized on CPU")

            return model, input_name

        except Exception as e:
            print(f"❌ Failed to initialize model on GPU: {e}")
            print("➡️ Falling back to CPUExecutionProvider")

            # Fallback to CPU only
            model = ort.InferenceSession(decrypted_model, providers=["CPUExecutionProvider"])
            input_name = model.get_inputs()[0].name
            return model, input_name


    def _inference(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Perform thread-safe inference on one image tile.

        Args:
            tile (torch.Tensor): Input tile tensor.

        Returns:
            torch.Tensor: Output tensor after super-resolution.
        """
        with self.lock:  # serialize inference calls for thread safety
            input_tile = {self.input_name: tile.numpy()}
            output_tile = self.network.run(None, input_tile)
        output_tensor = torch.from_numpy(output_tile[0])
        return output_tensor

    def run(self, img_np: np.ndarray) -> np.ndarray:
        """
        Run the super-resolution model on the full image.

        Args:
            img_np (np.ndarray): Input image RGB as numpy array.

        Returns:
            np.ndarray: Super-resolved output image as numpy uint8 array.
        """
        img_tiles, original_shape, padded_shape = self.dataloader.load_image(img_np)

        output_tiles = [self._inference(tile) for tile in img_tiles]

        output_img = self.dataloader.reconstruct_image_from_tiles_with_blending(
            output_tiles,
            padded_shape,
            self.scale,
        )

        # Crop to original size * scale
        output_img = output_img[
            :, : original_shape[0] * self.scale, : original_shape[1] * self.scale
        ]

        output_img = output_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        out_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)

        return out_img