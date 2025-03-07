import argparse
import json
import torch
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                            roc_auc_score, confusion_matrix,
                            precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from data.pannuke import PanNukeZeroShotDataset  
from data.puma import PUMAZeroShotDataset  
from data.transforms import create_zero_shot_transforms
from utils.config import load_config


def load_model(conch_path: str, fine_tuned_checkpoint: str = None, image_size: int = 224, device: str = 'cuda'):
    """Load CONCH model and add fine-tuned weights if provided.
    
    Args:
        conch_path: Path to the base CONCH model checkpoint
        fine_tuned_checkpoint: Optional path to fine-tuned weights
        image_size: Input image size
        device: Device to load the model on
    """
    model, preprocess = create_model_from_pretrained(
        model_cfg='conch_ViT-B-16',
        checkpoint_path=conch_path,  
        force_image_size=image_size,
        device=device
    )

    if fine_tuned_checkpoint:
        checkpoint = torch.load(fine_tuned_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, preprocess

def create_text_embeddings(class_names, descriptions_file, model, tokenizer, device):
    """Prepare text embeddings for each cell type using all available descriptions.
    
    For each cell type:
    1. Tokenize all descriptions
    2. Compute embeddings for each description
    3. Average embeddings to create an ensemble representation
    
    Args:
        class_names: List of class names
        descriptions_file: Path to the JSON file with class descriptions
        model: CONCH model
        tokenizer: Text tokenizer
        device: Device to run the model on
        
    Returns:
        torch.Tensor: Concatenated text embeddings for all classes
    """
    with open(descriptions_file, 'r') as f:
        cell_descriptions = json.load(f)

    text_embeddings = {}
    with torch.no_grad():
        for cell_type in class_names:
            desc = cell_descriptions[cell_type]                
            tokens = tokenize(texts=desc, tokenizer=tokenizer).to(device)
            embeddings = model.encode_text(tokens, normalize=True)
            text_embeddings[cell_type] = embeddings.mean(dim=0, keepdim=True)

    # Concatenate embeddings for the selected cell types, ready for cosine similarity
    return torch.cat([text_embeddings[ct] for ct in class_names])

@torch.no_grad()
def evaluate_zero_shot(model, test_loader, class_embeddings, class_names, device):
    """Use CONCH model to perform zero-shot classification on the provided dataset."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_similarity_scores = []

    # -- Zero-shot classification logic --
    for batch in test_loader:
        images = batch['image'].to(device)
        targets = batch['label'].to(device)
        
        image_embeddings = model.encode_image(images)
        
        # Cosine similarity between image and class embeddings
        similarity = image_embeddings @ class_embeddings.T
        
        predictions = similarity.argmax(dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_similarity_scores.extend(similarity.cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    similarity_scores = np.array(all_similarity_scores)
    
    metrics = compute_metrics(predictions, targets, similarity_scores, class_names)
    
    return metrics, predictions, targets

def compute_metrics(predictions, targets, similarity_scores, class_names):
    """Compute classification metrics."""
    
    global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    global_accuracy = balanced_accuracy_score(targets, predictions) 
    
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        targets, predictions, average=None, labels=range(len(class_names))
    )

    try:
        roc_auc = roc_auc_score(
            targets, 
            similarity_scores, 
            multi_class='ovo',
            average='macro'
        )
    except ValueError as e:
        print(f"Warning: Could not compute overall ROC AUC: {str(e)}")
        roc_auc = None

    metrics = {
        'accuracy': global_accuracy,
        'precision': global_precision,
        'recall': global_recall,
        'f1': global_f1,
        'roc_auc': roc_auc
    }

    metrics['per_class'] = {}
    for idx, class_name in enumerate(class_names):
        y_true = (targets == idx).astype(int)
        y_scores = similarity_scores[:, idx]

        try:
            roc_auc_per_class = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            print(f"Warning: Could not compute ROC AUC for {class_name}: {str(e)}")
            roc_auc_per_class = None

        metrics['per_class'][class_name] = {
            'precision': class_precision[idx],
            'recall': class_recall[idx],
            'f1': class_f1[idx],
            'roc_auc': roc_auc_per_class
        }
            
    return metrics

def save_evaluation_results(metrics, model_config, output_dir, experiment_name):
    """Save evaluation results and model configuration to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model_config': model_config,
        'metrics': metrics
    }
    
    if experiment_name:
        results['experiment_name'] = experiment_name
    
    timestamp = results['timestamp']
    filename = f"evaluation_results_{timestamp}.json"
    if experiment_name:
        filename = f"{experiment_name}_{filename}"
    
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_path}")

def save_confusion_matrix(confusion_matrix: np.ndarray, 
                         cell_types: List[str], 
                         save_path: Path):
    """Create and save a confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        confusion_matrix,
        xticklabels=cell_types,
        yticklabels=cell_types,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    config = load_config(args.config)
    device = config['device']

    model, _ = load_model(
        conch_path=config['model']['conch_path'],
        fine_tuned_checkpoint=config['model']['fine_tuned_checkpoint'],
        image_size=config['model']['image_size'],
        device=device
    )

    tokenizer = get_tokenizer()

    # Load corresponding dataset and apply transforms
    transform = create_zero_shot_transforms(image_size=config['model']['image_size'])
    if config['data']['dataset'] == 'pannuke':
        dataset = PanNukeZeroShotDataset(
            annotations_path=config['data']['annotations_path'],
            images_dir=config['data']['images_dir'],
            transform=transform
        )
        class_names = dataset.class_to_idx.keys()
        
    elif config['data']['dataset'] == 'puma':
        dataset = PUMAZeroShotDataset(
            annotations_path=config['data']['annotations_path'],
            images_dir=config['data']['images_dir'],
            transform=transform,
            selected_categories=config['data']['class_names']
        )
        class_names = dataset.categories
    else:
        raise ValueError("Invalid dataset name.  Must be 'pannuke' or 'puma'.")

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,  
        num_workers=config['num_workers'],
        pin_memory=True
    )

    class_embeddings = create_text_embeddings(
        class_names=class_names,
        descriptions_file=config['data']['descriptions_file'],
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    metrics, predictions, targets = evaluate_zero_shot(model, dataloader, class_embeddings, class_names, device)
    
    cm = confusion_matrix(targets, predictions)
    save_confusion_matrix(cm, class_names, Path(args.output_dir) / "confusion_matrix.png")
    
    save_evaluation_results(
        metrics=metrics,
        model_config={'model': 'conch_ViT-B-16', 'checkpoint': config['model']['fine_tuned_checkpoint']},  
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot cell classification with CONCH.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--experiment_name", type=str, default="zero_shot_eval", help="Name of the experiment for saving results.")
    parser.add_argument("--output_dir", type=str, default="results", help="Base directory to save evaluation results.")

    args = parser.parse_args()
    main(args)