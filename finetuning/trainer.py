import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from datetime import datetime
import json
from finetuning.metrics import compute_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR

class CONCHTrainer:
    """Trainer class for CONCH model fine-tuning.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config: Dict,
        device,
        experiments_base_dir,
        log_wandb: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = config['logging']['experiment_name']
        self.experiment_dir = Path(experiments_base_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self._save_config()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_log = []
        
        self.wandb = log_wandb
        if log_wandb:
            self._init_wandb()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch and compute all metrics."""
        self.model.train()
        total_loss = 0.0
        i2t_loss = 0.0
        t2i_loss = 0.0 
        batch_accuracy = 0.0
        image_features = []
        text_features = []
        cell_types = []
        
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            text_tokens = batch['description'].to(self.device)
            
            outputs = self.model(images, text_tokens)
            loss_dict = self.criterion(
                outputs['image_features'],
                outputs['text_features'],
                outputs['logit_scale']
            )
            
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            
            total_loss += loss_dict['loss'].item()
            i2t_loss += loss_dict['loss_i2t'].item()
            t2i_loss += loss_dict['loss_t2i'].item()
            batch_accuracy += loss_dict['accuracy'].item()
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'loss_i2t': f"{loss_dict['loss_i2t'].item():.4f}",
                'loss_t2i': f"{loss_dict['loss_t2i'].item():.4f}",
                'batch_accuracy': f"{loss_dict['accuracy'].item():.4f}"
            })
            
            image_features.append(outputs['image_features'].cpu().detach())
            text_features.append(outputs['text_features'].cpu().detach())
            cell_types.append(batch['cell_type'])
        
        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0) 
        cell_types = torch.cat(cell_types, dim=0)
        
        retrieval_metrics = compute_metrics(image_features, text_features, cell_types)
    
        avg_loss = total_loss / len(self.train_loader)
        avg_i2t_loss = i2t_loss / len(self.train_loader)
        avg_t2i_loss = t2i_loss / len(self.train_loader)
        batch_accuracy = batch_accuracy / len(self.train_loader)
        
        metrics = {
            'losses/train_loss': avg_loss,
            'losses/train_loss_i2t': avg_i2t_loss,
            'losses/train_loss_t2i': avg_t2i_loss,
            'train/batch_accuracy': batch_accuracy,
            **{f'train/{k}': v for k, v in retrieval_metrics.items()}
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute all metrics."""
        self.model.eval()
        total_loss = 0.0
        i2t_loss = 0.0
        t2i_loss = 0.0
        batch_accuracy = 0.0
        image_features = []
        text_features = []
        cell_types = []
        
        pbar = tqdm(self.val_loader,desc='Validation')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            text_tokens = self.model.tokenize(batch['description']).to(self.device)            
            outputs = self.model(images, text_tokens)
            loss_dict = self.criterion(
                outputs['image_features'],
                outputs['text_features'],
                outputs['logit_scale']
            )
            
            image_features.append(outputs['image_features'].cpu().detach())
            text_features.append(outputs['text_features'].cpu().detach())
            cell_types.append(batch['cell_type'])
            
            total_loss += loss_dict['loss'].item()
            i2t_loss += loss_dict['loss_i2t'].item()
            t2i_loss += loss_dict['loss_t2i'].item()
            batch_accuracy += loss_dict['accuracy'].item()
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'loss_i2t': f"{loss_dict['loss_i2t'].item():.4f}",
                'loss_t2i': f"{loss_dict['loss_t2i'].item():.4f}",
                'batch_accuracy': f"{loss_dict['accuracy'].item():.4f}"
            })
            
        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)
        cell_types = torch.cat(cell_types, dim=0)
    
        retrieval_metrics = compute_metrics(image_features, text_features, cell_types)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_i2t_loss = i2t_loss / len(self.val_loader)
        avg_t2i_loss = t2i_loss / len(self.val_loader)
        batch_accuracy = batch_accuracy / len(self.val_loader)
        
        metrics = {
            'losses/val_loss': avg_loss,
            'losses/val_loss_i2t': avg_i2t_loss,
            'losses/val_loss_t2i': avg_t2i_loss,
            'val/batch_accuracy': batch_accuracy,
            **{f'val/{k}': v for k, v in retrieval_metrics.items()}
        }
        
        return metrics

    def train(self, num_epochs: int):
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        
        # CosineAnnealingLR scheduler as in the original implementation of CONCH
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=0  
        )
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            # Validate every epoch (patience implemented correctly)
            val_metrics = self.validate()
            
            combined_metrics = {**train_metrics, **val_metrics}
            self._log_metrics(combined_metrics)
            
            if val_metrics['losses/val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['losses/val_loss']
                self.patience_counter = 0
                self._save_checkpoint(combined_metrics)
            else:
                self.patience_counter += 1
            
            if self.wandb:
                wandb.log({
                    **combined_metrics,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            scheduler.step()
            
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                print(f'Early stopping triggered after {epoch} epochs')
                break
            
        
        self._save_experiment_summary(combined_metrics)
            
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config['logging']['project_name'],
            name=self.config['logging']['experiment_name'],
            config=self.config
        )
    
    def _save_config(self):
        """Save experiment configuration to JSON file."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics for current epoch to JSON file."""
        epoch_log = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **metrics
        }
        self.training_log.append(epoch_log)
        
        log_path = self.experiment_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=4)
            
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint with unified naming scheme."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
    
        best_path = self.checkpoints_dir / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
            
    def _save_experiment_summary(self, final_metrics: Dict[str, float]):
        """Save final experiment summary including training history and best results."""
        best_metrics = {}
        best_val_loss = float('inf')
        for log in self.training_log:
            if log.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = log['val_loss']
                best_metrics = {k: v for k, v in log.items() 
                              if k not in ['epoch', 'timestamp']}

        summary = {
            'experiment_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'final_metrics': final_metrics,
            'best_metrics': best_metrics,
            'total_epochs': self.current_epoch,
            'early_stopped': self.patience_counter >= self.config['training']['patience']
        }
        
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)