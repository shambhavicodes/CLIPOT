import os
import torch
import argparse
from clip import clip
from modeling.models import Adapter
from sklearn.metrics import accuracy_score
from clipot import ClipOTLoss, ClipOTMethod
from utils.misc import set_global_seeds, save_configuration
from utils import datasets
import wandb
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from modeling.prompts import IMAGENET_TEMPLATES_SELECT
import numpy as np

def argparser():
    parser = argparse.ArgumentParser("ClipOT - Weight Averaged Adaptation of CLIP")

    # Directories
    parser.add_argument('--data_dir', type=str, default='data/', help='Root directory for datasets')
    parser.add_argument('--save_dir', type=str, default='save/', help='Path for saving results')

    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='Model backbone to use')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=(
        'cifar10', 'cifar100', 'tiny-imagenet', 'visda', 'PACS', 'office_home', 'VLCS', 'office31',
        'imagenet_r', 'imagenet_a', 'imagenet_v2', 'imagenet_sketch'
    ), help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--corruptions_list', nargs='+', default=[None], type=str, help='List of corruptions to apply to the dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')  # Added workers argument

    # ClipOT-specific arguments
    parser.add_argument('--temperature', type=float, default=0.01, help='Temperature for ClipOT loss')
    parser.add_argument('--epsilon', type=float, default=0.7, help='Epsilon for Sinkhorn-Knopp algorithm in ClipOT')
    parser.add_argument('--num_templates', type=int, default=8, help='Number of templates to use')
    parser.add_argument('--use_avg_embeddings', action='store_true', help='Use averaged text embeddings')

    # Logging
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')

    return parser


def main():
    args = argparser().parse_args()

    set_global_seeds(args.seed)
    save_configuration(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model
    model_clip, clip_transforms = clip.load(args.backbone, device=device)

    # Store overall results
    overall_results = {}
    correct = 0
    total = 0

    # Loop through corruptions
    for corruption in args.corruptions_list:
        if corruption:
            corruption_name = corruption
        else:
            corruption_name = "No_Corruption"

        # Prepare dataset
        data_loader, classnames = datasets.prepare_data(
            dataset=args.dataset,
            data_dir=args.data_dir,
            corruption=corruption,
            batch_size=args.batch_size,
            num_workers=args.workers,  
            clip_transforms=clip_transforms
        )

        # Prepare the adapter
        model = Adapter(model_clip.to(torch.float32), classnames=classnames, templates=IMAGENET_TEMPLATES_SELECT)
        model = model.to(device)

        if args.use_avg_embeddings:
            prototypes = model.text_embeddings_avg.to(device)
        else:
            prototypes = model.text_embeddings.to(device)

        # Initialize ClipOT method
        clipot_loss = ClipOTLoss(prototypes=prototypes, temperature=args.temperature, epsilon=args.epsilon)
        adapt_method = ClipOTMethod(
            model=model,
            clipot_loss=clipot_loss,
            text_embeddings=model.text_embeddings.to(device),
            use_avg_embeddings=args.use_avg_embeddings,
            lr=1e-4
        )

        if not args.disable_wandb:
            wandb.init(project=f"ClipOT-{args.dataset}-{corruption_name}", config=vars(args))

        results = []
        with tqdm(total=len(data_loader), desc=f"Processing Batches for Corruption: {corruption_name}") as pbar:
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Adaptation phase
                total_loss = 0.0
                template_indices = list(range(args.num_templates))
                for i in template_indices:
                    adapt_method.model.adapter.template_index = i
                    adapt_method.model.adapter.prototypes.data = adapt_method.text_embeddings[:, i, :].to(device)

                    # Update prototypes and compute ClipOT loss
                    adapt_method.clipot_loss.prototypes = adapt_method.text_embeddings[:, i, :].to(device)
                    logits, img_feats = adapt_method.model(inputs)
                    loss_clipot = adapt_method.clipot_loss(img_feats, logits)

                    loss_clipot.backward()
                    adapt_method.optimizer.step()
                    adapt_method.optimizer.zero_grad()

                # Compute averaged prototype
                avg_prototype = adapt_method.text_embeddings.mean(dim=1).to(device)
                adapt_method.model.adapter.prototypes.data = avg_prototype

                # Evaluation phase with averaged prototypes
                with torch.no_grad():
                    logits, _ = adapt_method.model(inputs)
                preds = torch.argmax(logits, dim=1)

                # Calculate accuracy
                batch_acc = np.round(accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()) * 100, 2)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                results.append(batch_acc)
                pbar.set_postfix(batch_accuracy=batch_acc)
                pbar.update(1)

                if not args.disable_wandb:
                    wandb.log({"Batch Accuracy": batch_acc})

        if not args.disable_wandb:
            wandb.finish()
        acc = (correct / total) * 100 if total > 0 else 0
        print(f"Accuracy for Corruption '{corruption_name}': {acc:.2f}%")


if __name__ == "__main__":
    main()
