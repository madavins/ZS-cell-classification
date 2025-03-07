import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path

class PanNukeZeroShotDataset(Dataset):
    def __init__(self, annotations_path, images_dir, transform=None):
        """
        Dataset for loading PanNuke images for zero-shot evaluation.

        Args:
            annotations_file: Path to the JSON file containing image metadata
                               (file_name, cell_type).  This is different
                               from the fine-tuning annotations.
            images_dir: Path to the directory containing the cropped images.
            transform: Optional transform to be applied to the image.
        """
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)['annotations'] 
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_to_idx = {'inflammatory': 0, 'connective': 1, 'necrosis': 2, 'epithelial': 3}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = self.images_dir / ann['file_name']
        image = Image.open(image_path).convert('RGB')
        label = self.class_to_idx[ann['cell_type']] 

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'cell_type': ann['cell_type'], 
            'file_name': ann['file_name']  
        }