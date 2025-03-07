import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union
from enum import Enum
from pathlib import Path
from conch.open_clip_custom import create_model_from_pretrained

class FineTuneStage(Enum):
    """Enumeration of fine-tuning stages."""
    POOLER_ONLY = "pooler_only"        # Stage 1: Only fine-tune attentional pooler
    TOP_LAYERS = "top_layers"          # Stage 2: Fine-tune top layers and pooler
    FULL_ENCODER = "full_encoder"      # Stage 3: Fine-tune entire vision encoder
    TEXT_VISION_v1 = "text_vision_v1"  # Stage 4: Fine-tune text projection and attentional pooler
    BOTH_ENCODERS = "both_encoders"    # Stage 5: Fine-tune entire vision and text encoders

class CONCHModel(nn.Module):
    """CONCH model adapted for cell classification.
    
    This class handles:
    1. Loading and initialization of CONCH
    2. Progressive fine-tuning stages
    3. Forward pass configurations
    4. Feature extraction and embedding computation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        stage: str,
        num_top_layers: int = 2,
        device: Optional[str] = None,
        image_size: int = 112
    ):
        """Initialize CONCH model.
        
        Args:
            checkpoint_path: Path to the CONCH checkpoint (.bin file)
            stage: Fine-tuning stage ('pooler_only', 'top_layers', or 'full_encoder')
            num_top_layers: Number of top layers to fine-tune in TOP_LAYERS stage
            device: Device to load model on (default: cuda if available, else cpu)
            image_size: Input image size for the model
        """
        super().__init__()
        
        self.device = device
        
        self.model, self.preprocess = self._load_model(checkpoint_path, image_size)
        self.model = self.model.to(self.device)
        
        # Set stage and configure fine-tuning
        self.stage = FineTuneStage(stage)
        self.num_top_layers = num_top_layers
        self._configure_fine_tuning()
        
    def _load_model(self, checkpoint_path: str, image_size: int):
        """Load pretrained CONCH model.
        
        Args:
            checkpoint_path: Path to CONCH original checkpoint (.bin file)
            image_size: Input image size
            
        Returns:
            tuple: (model, preprocess_function)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        try:
            model, preprocess = create_model_from_pretrained(
                model_cfg='conch_ViT-B-16',
                checkpoint_path=str(checkpoint_path),
                force_image_size=image_size
            )
            model.eval()
            return model, preprocess
        except Exception as e:
            raise RuntimeError(f"Error loading CONCH model: {str(e)}")
    
    def _configure_fine_tuning(self):
        """Configure model parameters based on current fine-tuning stage."""
        # First, freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
            
        if self.stage == FineTuneStage.POOLER_ONLY:
            # Unfreeze only attentional pooler
            for param in self.model.visual.attn_pool_contrast.parameters():
                param.requires_grad = True
                
        elif self.stage == FineTuneStage.TOP_LAYERS:
            # Unfreeze pooler and top transformer layers
            for param in self.model.visual.attn_pool_contrast.parameters():
                param.requires_grad = True
                
            # Unfreeze top layers of vision encoder
            vision_layers = self.model.visual.trunk.blocks
            for layer in vision_layers[-self.num_top_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        elif self.stage == FineTuneStage.FULL_ENCODER:
            # Unfreeze entire vision encoder
            for param in self.model.visual.parameters():
                param.requires_grad = True
                
        elif self.stage == FineTuneStage.TEXT_VISION_v1:
            # Unfreeze attentional pooler and text projection
            for param in self.model.visual.attn_pool_contrast.parameters():
                param.requires_grad = True
            self.model.text.text_projection.requires_grad = True
                
        elif self.stage == FineTuneStage.BOTH_ENCODERS:
            # Unfreeze entire vision and text encoders
            for param in self.model.visual.parameters():
                param.requires_grad = True
            for param in self.model.text.parameters():
                param.requires_grad = True  
    
    def forward(self, 
                images: torch.Tensor, 
                text_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Get image features
        image_features = self.model.encode_image(images, normalize=True, proj_contrast=True)
        
        outputs = {'image_features': image_features}
        
        # Get text features if provided
        if text_tokens is not None:
            text_features = self.model.encode_text(text_tokens, normalize=True)
            outputs['text_features'] = text_features
        
        # Add temperature parameter
        outputs['logit_scale'] = self.model.logit_scale.exp()
        
        return outputs
    
    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode images using the CONCH vision encoder."""
        return self.model.encode_image(images, normalize=normalize)

    def encode_text(self, text_tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode text using the CONCH text encoder."""
        return self.model.encode_text(text_tokens, normalize=normalize)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of currently trainable parameters based on stage."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_stage(self) -> FineTuneStage:
        """Get current fine-tuning stage."""
        return self.stage
    
    def set_stage(self, stage: str):
        """Update fine-tuning stage and reconfigure model."""
        self.stage = FineTuneStage(stage)
        self._configure_fine_tuning()