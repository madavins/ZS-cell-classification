import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path

class PUMAZeroShotDataset(Dataset):
    def __init__(self, annotations_path, images_dir, transform=None, selected_categories=None):
      """
      Dataset for loading PUMA images for zero-shot evaluation.

      Args:
          annotations_file: Path to the JSON file containing image metadata
                             (file_name, category_id).
          images_dir: Path to the directory containing the cropped images.
          transform: Optional transform to be applied to the image.
          selected_categories: list of the selected categories by name (e.g. ['lymphocite', 'tumor])
      """
      with open(annotations_path, 'r') as f:
          data = json.load(f)
          self.annotations = data['annotations']
          categories = sorted(data['categories'], key=lambda x: x['id']) #must be sorted by id
          
      self.images_dir = Path(images_dir)
      self.transform = transform
      
      if selected_categories:
          self.categories = [c['name'] for c in categories if c['name'] in selected_categories]
      else:
          self.categories = [c['name'] for c in categories]
      
      self.cat_to_id = {cat: i for i, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
      ann = self.annotations[idx]
      image_path = self.images_dir / ann['file_name']
      image = Image.open(image_path).convert('RGB')
      
      category_id = ann['category_id']
      category = self.categories[category_id]
      label = self.cat_to_id[category]

      if self.transform:
          image = self.transform(image)

      return {
          'image': image,
          'label': label,
          'cell_type': category,  
          'file_name': ann['file_name']  
      }