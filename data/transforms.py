from torchvision import transforms
from torchvision.transforms import InterpolationMode

def create_zero_shot_transforms(image_size=224):
    """
    Create data transformations for zero-shot evaluation (no augmentation).
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform