from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


class SA_Tiling_ImageLoader:
    """
    A utility class for loading, processing, and saving images. It supports operations such as loading images from disk,
    splitting images into tiles (with or without overlap), reconstructing images from tiles, applying linear weighting
    for blending, and saving the processed images back to disk.
    """

    def __init__(self, tile_size: int):
        """
        Initializes the ImageLoader with a specific tile size.

        Args:
            tile_size (int): The size of the squared tiles into which the images will be split. Defaults to 128.
        """
        self.tile_size = tile_size


        """
        Loads an image from a given path, optionally adds padding if either dimension is smaller than self.tile_size,
        splits the possibly padded image into overlapped tiles, and converts them to tensor format.

        Parameters:
            image_np (np.ndarray): The numpy array representing the image in RGB format.

        Returns:
            Tuple[List[torch.Tensor], Tuple[int, int], Tuple[int, int]]: A tuple containing a list of image tiles as torch tensors,
            the original image shape (height, width), and the shape after padding (height, width) if padding was applied.
            If no padding was applied, the original image shape and the shape after padding will be the same.
        """
    
    def load_image(
        self, image_np: np.ndarray
    ) -> Tuple[List[torch.Tensor], Tuple[int, int], Tuple[int, int]]:

        if isinstance(image_np, torch.Tensor):
            image_np = image_np.detach().cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))  # CHW → HWC

        # ✅ Assicurati che image_np sia in HWC (altezza, larghezza, canali)
        if image_np.ndim == 3 and image_np.shape[2] != 3:
            raise ValueError(f"Expected image in HWC format, got shape {image_np.shape}")

        original_shape = (image_np.shape[0], image_np.shape[1])

        padding_needed = (
            original_shape[0] < self.tile_size or original_shape[1] < self.tile_size
        )
        padded_shape = (
            max(original_shape[0], self.tile_size),
            max(original_shape[1], self.tile_size),
        )

        if padding_needed:
            padded_image_np = np.zeros(
                (padded_shape[0], padded_shape[1], 3), dtype=np.uint8
            )
            # ✅ Questa operazione ora ha senso perché image_np è HWC
            padded_image_np[: original_shape[0], : original_shape[1]] = image_np
        else:
            padded_image_np = image_np

        # ✅ Converti in formato CHW per PyTorch
        padded_image_for_tensor = (
            padded_image_np.astype(np.float32).transpose([2, 0, 1]) / 255.0
        )
        tiles = self.split_to_tiles_with_overlap(padded_image_for_tensor)
        tiles_tensor = [
            torch.as_tensor(tile[None, :, :, :], dtype=torch.float32) for tile in tiles
        ]

        return tiles_tensor, original_shape, padded_shape


    def split_to_tiles_with_overlap(
        self, image: np.ndarray, overlap_fraction: float = 0.25
    ) -> List[np.ndarray]:
        """
        Splits an image into overlapping tiles. Overlap between tiles is controlled by the overlap_fraction parameter.

        Parameters:
            image (np.ndarray): The input image as a NumPy array with shape (C, H, W).
            overlap_fraction (float, optional): The fraction of the tile size that each tile overlaps with its neighbors. Defaults to 0.25.

        Returns:
            List[np.ndarray]: A list of image tiles, each as a NumPy array with shape (C, tile_size, tile_size).
        """
        C, H, W = image.shape
        overlap = int(self.tile_size * overlap_fraction)
        stride = self.tile_size - overlap

        num_tiles_h = np.ceil((H - overlap) / stride).astype(int)
        num_tiles_w = np.ceil((W - overlap) / stride).astype(int)

        tiles = []

        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                start_h = h * stride
                start_w = w * stride

                if h == num_tiles_h - 1:
                    start_h = max(H - self.tile_size, 0)
                if w == num_tiles_w - 1:
                    start_w = max(W - self.tile_size, 0)

                end_h = start_h + self.tile_size
                end_w = start_w + self.tile_size

                tile = image[:, start_h:end_h, start_w:end_w]
                tiles.append(tile)

        return tiles

    def reconstruct_image_from_tiles_with_blending(
        self,
        upscaled_tiles: List[torch.Tensor],
        original_shape: Tuple[int, int],
        scale: int,
        overlap_percentage: float = 0.25,
    ) -> torch.Tensor:
        """
        Reconstructs an image from its upscaled tiles with blending to minimize seams.

        Parameters:
            upscaled_tiles (List[torch.Tensor]): A list of upscaled image tiles.
            original_shape (Tuple[int, int]): The height and width of the original image.
            scale (int): The factor by which the image has been upscaled.
            overlap_percentage (float, optional): The percentage of each tile that overlaps with its neighbors. Defaults to 0.25.

        Returns:
            torch.Tensor: The reconstructed and blended upscaled image.
        """
        upscaled_tile_size = self.tile_size * scale
        overlap = int(self.tile_size * overlap_percentage)
        stride = self.tile_size - overlap
        upscaled_stride = stride * scale

        H, W = original_shape
        upscaled_H = H * scale
        upscaled_W = W * scale

        upscaled_image = torch.zeros((3, upscaled_H, upscaled_W))
        weight_map = torch.zeros((3, upscaled_H, upscaled_W))

        upscaled_overlap = overlap * scale

        num_tiles_h = np.ceil((H - overlap) / stride).astype(int)
        num_tiles_w = np.ceil((W - overlap) / stride).astype(int)

        tile_index = 0
        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                start_h = h * upscaled_stride
                start_w = w * upscaled_stride

                if h == num_tiles_h - 1:
                    start_h = upscaled_H - upscaled_tile_size
                if w == num_tiles_w - 1:
                    start_w = upscaled_W - upscaled_tile_size

                tile = upscaled_tiles[tile_index]
                tile_index += 1

                # Calcola i pesi per il blending
                weight = np.ones(
                    (upscaled_tile_size, upscaled_tile_size), dtype=np.float32
                )
                if h > 0:  # Sovrapposizione superiore
                    weight[:upscaled_overlap, :] *= np.outer(
                        self.linear_weight(
                            np.arange(upscaled_overlap), upscaled_overlap
                        ),
                        np.ones(upscaled_tile_size),
                    )
                if w > 0:  # Sovrapposizione a sinistra
                    weight[:, :upscaled_overlap] *= np.outer(
                        np.ones(upscaled_tile_size),
                        self.linear_weight(
                            np.arange(upscaled_overlap), upscaled_overlap
                        ),
                    )
                if h < num_tiles_h - 1:  # Sovrapposizione inferiore
                    weight[-upscaled_overlap:, :] *= np.outer(
                        self.linear_weight(
                            np.arange(upscaled_overlap - 1, -1, -1), upscaled_overlap
                        ),
                        np.ones(upscaled_tile_size),
                    )
                if w < num_tiles_w - 1:  # Sovrapposizione a destra
                    weight[:, -upscaled_overlap:] *= np.outer(
                        np.ones(upscaled_tile_size),
                        self.linear_weight(
                            np.arange(upscaled_overlap - 1, -1, -1), upscaled_overlap
                        ),
                    )

                weight_tensor = torch.from_numpy(weight).float()
                weight_tensor = weight_tensor.unsqueeze(0)
                weight_tensor = weight_tensor.expand(3, -1, -1)
                tile = tile.squeeze(0)

                upscaled_image[
                    :,
                    start_h : start_h + upscaled_tile_size,
                    start_w : start_w + upscaled_tile_size,
                ] += (
                    tile * weight_tensor
                )

                weight_map[
                    :,
                    start_h : start_h + upscaled_tile_size,
                    start_w : start_w + upscaled_tile_size,
                ] += weight_tensor

        # Normalizza l'immagine finale dividendo per il weight_map
        upscaled_image /= weight_map

        return upscaled_image

    def linear_weight(self, x: np.ndarray, width: float) -> np.ndarray:
        """
        Generates a linear weight that increases from 0 to 1.

        Parameters:
            x (np.ndarray): The input array.
            width (float): The width over which to apply the linear weight.

        Returns:
            np.ndarray: The linearly weighted array.
        """
        return np.minimum(x / width, (width - x) / width)

    def save_image(
        self, pred: np.ndarray, scale: int, out_path: str, dpi: int = 300
    ) -> None:
        """
        Saves the predicted image tensor to a specified file path.

        Parameters:
            pred (np.ndarray): The predicted image array.
            out_path (str): The output path where the image will be saved.
            dpi (int): Image DPI
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pil_image = Image.fromarray(pred)

        pil_image.save(out_path, dpi=(dpi * scale, dpi * scale))