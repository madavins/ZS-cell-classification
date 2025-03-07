import argparse
import torch
from torch.utils.data import DataLoader
from utils.config import load_config 
from finetuning.data.dataset import PanNukeFineTuningDataset
from finetuning.data.transforms import create_finetuning_transforms
from finetuning.models.conch import CONCHModel
from finetuning.models.losses import ContrastiveLoss
from finetuning.trainer import CONCHTrainer

def main(config_path: str):
    """
    Main function to manage the retraining of the CONCH model for cell classification.
    """
    config = load_config(config_path)
    device = config['training']['device']
    
    train_transform, val_transform = create_finetuning_transforms(
        image_size=config['model']['image_size']
    )
    
    train_dataset = PanNukeFineTuningDataset(
        images_dir=config['data']['train_images_dir'],
        annotations_file=config['data']['train_annotations'],
        transform=train_transform
    )
    
    val_dataset = PanNukeFineTuningDataset(
        images_dir=config['data']['val_images_dir'],
        annotations_file=config['data']['val_annotations'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    model = CONCHModel(
        checkpoint_path=config['model']['checkpoint_path'],
        stage=config['model']['stage'],
        num_top_layers=config['model']['num_top_layers'],
        device=device,
        image_size=config['model']['image_size']
    )
    
    criterion = ContrastiveLoss()
    
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    trainer = CONCHTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config, 
        device=device,
        experiments_base_dir=config['logging']['experiments_dir']
    )
    
    trainer.train(num_epochs=config['training']['num_epochs'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CONCH model for cell classification')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)