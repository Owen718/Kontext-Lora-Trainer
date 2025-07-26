#import webdataset as wds
import torch 
import numpy as np
from torchvision import transforms
import math
import json
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.utils import save_image
import glob
from typing import List, Dict, Any, Optional
import re
import torchvision.transforms.functional as F
import random





# Taken from https://github.com/tmbdev-archive/webdataset-imagenet-2/blob/01a4ab54307b9156c527d45b6b171f88623d2dec/imagenet.py#L65.
def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
    else:
        yield from src

def collate_fn(samples):
    source_pixel_values = torch.stack([example["source_pixel_values"] for example in samples])
    source_pixel_values = source_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in samples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [example["prompt"] for example in samples]
    return {"source_pixel_values": source_pixel_values, "edited_pixel_values": edited_pixel_values, "captions": captions}

        
import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def paired_augmentation(src_img, tar_img):
    # Random horizontal flip
    if random.random() < 0.5:
        src_img = F.hflip(src_img)
        tar_img = F.hflip(tar_img)
    # Random rotation
    angle = random.uniform(-30, 30)  # Rotation angle range
    src_img = F.rotate(src_img, angle)
    tar_img = F.rotate(tar_img, angle)
    # Random rescale and crop
    scale = random.uniform(0.7, 1.3)
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        src_img, scale=(scale, scale), ratio=(1.0, 1.0)
    )
    src_img = F.resized_crop(src_img, i, j, h, w, size=(1024, 1024))
    tar_img = F.resized_crop(tar_img, i, j, h, w, size=(1024, 1024))
    return src_img, tar_img

class OmniConsistencyDataset(Dataset):
    """
    OmniConsistency Dataset Class - for multi-style image translation tasks.
    
    This dataset contains 22 different art styles, each sample includes:
    - src: original image
    - tar: stylized image
    - prompt: text describing the style
    """
    
    # List of all supported styles
    SUPPORTED_STYLES = [
        "3D_Chibi", "American_Cartoon", "Chinese_Ink", "Clay_Toy", "Fabric",
        "Ghibli", "Irasutoya", "Jojo", "LEGO", "Line", "Macaron", "Oil_Painting",
        "Origami", "Paper_Cutting", "Picasso", "Pixel", "Poly", "Pop_Art",
        "Rick_Morty", "Snoopy", "Van_Gogh", "Vector"
    ]
    
    def __init__(
        self,
        root_dir: str,
        styles: Optional[List[str]] = None,
        resolution: int = 512,
        split: str = "train",
        max_samples_per_style: Optional[int] = None,
        validate_images: bool = True,
        cache_images: bool = False,
        only_style_type_as_prompt: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: The root directory of the dataset.
            styles: A list of styles to load. Loads all styles if None.
            resolution: Image resolution.
            split: Data split ("train", "val", "test").
            max_samples_per_style: Maximum number of samples per style.
            validate_images: Whether to validate the existence of image files.
            cache_images: Whether to cache images in memory.
            random_seed: Random seed.
        """
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.split = split
        self.max_samples_per_style = max_samples_per_style
        self.validate_images = validate_images
        self.cache_images = cache_images
        self.only_style_type_as_prompt = only_style_type_as_prompt
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            
        # Determine which styles to load
        self.styles = styles if styles is not None else self.SUPPORTED_STYLES
        self._validate_styles()
        
        # Initialize image transformations
        self.image_transforms = transforms.Compose([
            # transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        
        # Load data
        self.samples = self._load_samples()
        
        # Cache images if enabled
        self.image_cache = {}
        if self.cache_images:
            self._cache_images()
            
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.styles)} styles.")
    
    def _validate_styles(self):
        """Validate the list of styles."""
        invalid_styles = set(self.styles) - set(self.SUPPORTED_STYLES)
        if invalid_styles:
            raise ValueError(f"Unsupported styles: {invalid_styles}")
            
        # Check if style directories exist
        missing_dirs = []
        for style in self.styles:
            style_dir = self.root_dir / style
            if not style_dir.exists():
                missing_dirs.append(style)
        
        if missing_dirs:
            raise FileNotFoundError(f"Missing style directories: {missing_dirs}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all sample data."""
        samples = []
        
        for style in self.styles:
            style_dir = self.root_dir / style
            jsonl_file = style_dir / f"{self.split}.jsonl"
            
            if not jsonl_file.exists():
                logger.warning(f"File not found: {jsonl_file}")
                continue
                
            style_samples = self._load_style_samples(jsonl_file, style)
            
            # Limit the number of samples per style
            if self.max_samples_per_style is not None and self.max_samples_per_style > 0:
                style_samples = style_samples[:self.max_samples_per_style]
                
            samples.extend(style_samples)
            logger.info(f"Loaded {len(style_samples)} samples from {style}")
        
        if not samples:
            raise ValueError("No valid samples were found.")
            
        return samples
    
    def _load_style_samples(self, jsonl_file: Path, style: str) -> List[Dict[str, Any]]:
        """Load samples for a single style."""
        samples = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    sample = {
                        'src_path': self.root_dir / data['src'],
                        'tar_path': self.root_dir / data['tar'],
                        'prompt': data['prompt'],
                        'style_type': style,
                        'line_num': line_num
                    }
                    
                    # Validate image file existence if enabled
                    if self.validate_images:
                        if not sample['src_path'].exists():
                            logger.warning(f"Source image missing: {sample['src_path']}")
                            continue
                        if not sample['tar_path'].exists():
                            logger.warning(f"Target image missing: {sample['tar_path']}")
                            continue
                    
                    samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in {jsonl_file}:{line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error loading sample at {jsonl_file}:{line_num}: {e}")
        
        return samples
    
    def _cache_images(self):
        """Cache all images into memory."""
        logger.info("Starting to cache images...")
        for i, sample in enumerate(self.samples):
            try:
                src_img = Image.open(sample['src_path']).convert('RGB')
                tar_img = Image.open(sample['tar_path']).convert('RGB')
                
                self.image_cache[i] = {
                    'src': src_img,
                    'tar': tar_img
                }
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Cached {i + 1}/{len(self.samples)} images.")
                    
            except Exception as e:
                logger.error(f"Failed to cache image {sample['src_path']}: {e}")
        
        logger.info("Image caching complete.")
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a solid color image as a fallback
            return Image.new('RGB', (self.resolution, self.resolution), color=(128, 128, 128))
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict containing:
                - img: edited_pixel_values (stylized image)
                - txt: prompt (text description)
                - ref_imgs: source_pixel_values (original image)
                - style_type: style_type (style type)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        if self.only_style_type_as_prompt:
            prompt = "Turn this image into the " + sample['style_type'].replace("_", " ") + " style"
        else:
            prompt = sample['prompt']
        try:
            # Load images from cache or file
            if self.cache_images and idx in self.image_cache:
                src_img = self.image_cache[idx]['src']
                tar_img = self.image_cache[idx]['tar']
            else:
                src_img = self._load_image(sample['src_path'])
                tar_img = self._load_image(sample['tar_path'])
            

            src_img, tar_img = paired_augmentation(src_img, tar_img)
            # Apply image transformations
            source_pixel_values = self.image_transforms(src_img)
            edited_pixel_values = self.image_transforms(tar_img)
            
            return {
                "img": edited_pixel_values,
                "txt": prompt,
                "ref_imgs": source_pixel_values,
                "style_type": sample['style_type']
            }
            
        except Exception as e:
            logger.error(f"Error getting sample {idx}: {e}")
            # Return a default sample on error
            return self._get_default_sample()
    
    def _get_default_sample(self) -> Dict[str, torch.Tensor]:
        """Return a default sample (used on error)."""
        default_img = torch.zeros(3, self.resolution, self.resolution)
        return {
            "img": default_img,
            "txt": "Default sample",
            "ref_imgs": default_img,
            "style_type": "unknown"
        }
    
    def get_style_info(self) -> Dict[str, int]:
        """Get statistics of sample counts for each style."""
        style_counts = {}
        for sample in self.samples:
            style = sample['style_type']
            style_counts[style] = style_counts.get(style, 0) + 1
        return style_counts
    
    def get_sample_by_style(self, style: str, limit: Optional[int] = None) -> List[int]:
        """Get sample indices for a specific style."""
        if style not in self.styles:
            raise ValueError(f"Style '{style}' is not in the list of loaded styles.")
        
        indices = [i for i, sample in enumerate(self.samples) 
                  if sample['style_type'] == style]
        
        if limit is not None:
            indices = indices[:limit]
            
        return indices
    
    def shuffle(self, seed: Optional[int] = None):
        """Shuffle the dataset randomly."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.samples)
        
        # If caching is enabled, the cache needs to be rebuilt
        if self.cache_images:
            self._cache_images()
    
    def save_samples_info(self, output_file: str):
        """Save sample information to a file."""
        info = {
            'total_samples': len(self.samples),
            'styles': self.styles,
            'style_counts': self.get_style_info(),
            'resolution': self.resolution,
            'split': self.split
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sample information has been saved to {output_file}")


# Convenience function
def create_omni_consistency_dataset(
    root_dir: str,
    styles: Optional[List[str]] = None,
    resolution: int = 512,
    split: str = "train",
    **kwargs
) -> OmniConsistencyDataset:
    """
    Convenience function to create an OmniConsistencyDataset.
    
    Args:
        root_dir: The root directory of the dataset.
        styles: A list of styles to load.
        resolution: Image resolution.
        split: Data split.
        **kwargs: Other arguments.
    
    Returns:
        An instance of OmniConsistencyDataset.
    """
    return OmniConsistencyDataset(
        root_dir=root_dir,
        styles=styles,
        resolution=resolution,
        split=split,
        **kwargs
    )

