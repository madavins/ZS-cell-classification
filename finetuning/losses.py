import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning image and text embeddings.
    
    This implements the bidirectional contrastive loss used in original implementation of CONCH/CoCa:
    L = (L_i2t + L_t2i) / 2
    where:
    - L_i2t is image-to-text contrastive loss
    - L_t2i is text-to-image contrastive loss
    """
    
    def forward(self, image_features: torch.Tensor, 
                text_features: torch.Tensor, 
                logit_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss.
        
        Args:
            image_features: Normalized image embeddings [batch_size, embed_dim]
            text_features: Normalized text embeddings [batch_size, embed_dim]
            logit_scale: Learned parameter (1/temperature)
            
        Returns:
            Dictionary containing:
                - loss: Total contrastive loss
                - i2t_loss: Image-to-text loss
                - t2i_loss: Text-to-image loss
                - accuracy: Percentage of correct matches
        """
        batch_size = image_features.shape[0]
        
        # Similarity matrix
        logits = (image_features @ text_features.T) * logit_scale
    
        # Labels are diagonal -> we indicate the "correct class" for each sample
        labels = torch.arange(batch_size, device=image_features.device)
        
        i2t_loss = F.cross_entropy(logits, labels)
        t2i_loss = F.cross_entropy(logits.T, labels)
        
        # Total loss is average of both directions
        total_loss = (i2t_loss + t2i_loss) / 2
        
        with torch.no_grad():
            i2t_pred = logits.argmax(dim=1)
            t2i_pred = logits.argmax(dim=0)
            i2t_acc = (i2t_pred == labels).float().mean()
            t2i_acc = (t2i_pred == labels).float().mean()
            accuracy = (i2t_acc + t2i_acc) / 2
        
        return {
            'loss': total_loss,
            'loss_i2t': i2t_loss,
            'loss_t2i': t2i_loss,
            'accuracy': accuracy
        }