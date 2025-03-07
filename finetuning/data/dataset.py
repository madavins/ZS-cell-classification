import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from conch.open_clip_custom import get_tokenizer, tokenize

class PanNukeFineTuningDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        """
        Dataset for loading PanNuke images and tokenizing descriptions for fine-tuning.

        Args:
            annotations_file: Path to the JSON file containing annotations
                               with 'file_name' and 'description' fields.
            images_dir: Path to the directory containing the cropped cell images.
            transform: Optional transform to be applied to the image.
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = self.images_dir / ann['file_name']

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # Text description is directly tokenized
            description = ann['description']
            tokens = tokenize(self.tokenizer, [description]).squeeze(0)

            return {
                'image': image,
                'description': tokens,
                'text_description': description,
                'cell_type': ann['cell_type']
            }

        except (IOError, OSError) as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise RuntimeError(f"Failed to load image at {image_path}") from e
        except Exception as e:
            print(f"Unexpected error processing item {idx}: {str(e)}")
            raise RuntimeError(f"Failed to process dataset item {idx}") from e

    def get_cell_type_distribution(self):
        """Returns the distribution of cell types in the dataset."""
        from collections import Counter
        cell_types = [ann['cell_type'] for ann in self.annotations]
        return Counter(cell_types)