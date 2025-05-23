

# Words Matter: Leveraging Individual Text Embeddings for Code Generation in CLIP Test-Time Adaptation

---

## Setup Instructions

### Prerequisites

1. **Python**: Requires Python >= 3.8.
2. **PyTorch**: Compatible with PyTorch >= 1.10.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repository/clipot.git
   cd CLIPOT
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv clipot_env
   source clipot_env/bin/activate  # On Windows use `clipot_env\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running CLIP-OT

1. **Prepare the dataset**:

   Place your dataset in the `data/` directory. For example, for CIFAR10:

   ```
   data/
   └── CIFAR10/
       ├── train/
       └── test/
   ```

2. **Run the script**:

   Use the provided `run.sh` script to execute the training:

   ```bash
   bash run.sh
   ```

   The script includes the following configurable parameters:

   - `DATA_DIR`: Path to your dataset.
   - `SAVE_DIR`: Directory to save results.
   - `BATCH_SIZE`: Batch size for training and inference.
   - `EPSILON`: Regularization parameter for Sinkhorn-Knopp algorithm.
   - `NUM_TEMPLATES`: Number of templates for adaptation.

### Example Command

To run CLIP-OT on CIFAR10 without corruption logging:

```bash
CUDA_VISIBLE_DEVICES=4 python main.py \
    --dataset cifar10 \
    --data_dir ./data/CIFAR10 \
    --save_dir ./save \
    --backbone ViT-B/32 \
    --batch-size 128 \
    --epsilon 0.7 \
    --num_templates 8 \
    --corruptions_list original \
    --disable_wandb
```

---

## Directory Structure

- **`main.py`**: Main script to execute CLIP-OT.
- **`run.sh`**: Bash script for running experiments with pre-defined configurations.
- **`clip/`**: Contains the implementation of CLIP and related utilities.
- **`utils/`**: Includes helper functions for data preparation and logging.
- **`modeling/`**: Contains model adapters and template definitions.

---

## Arguments and Options

| Argument               | Default     | Description                                                                 |
|------------------------|-------------|-----------------------------------------------------------------------------|
| `--data_dir`           | `data/`     | Root directory for datasets.                                                |
| `--save_dir`           | `save/`     | Path to save results.                                                       |
| `--seed`               | `42`        | Random seed for reproducibility.                                            |
| `--backbone`           | `ViT-B/32`  | Model backbone to use (CLIP variants).                                      |
| `--dataset`            | `cifar10`   | Dataset to use. Options: CIFAR10, CIFAR100, PACS, OfficeHome, TinyImageNet, etc.|
| `--batch-size`         | `128`       | Batch size for training and evaluation.                                     |
| `--corruptions_list`   | `[None]`    | List of corruptions to apply to the dataset.                                |
| `--workers`            | `4`         | Number of workers for data loading.                                         |                                             |
| `--epsilon`            | `0.7`       | Regularization parameter for Sinkhorn-Knopp algorithm.                      |
| `--num_templates`      | `8`         | Number of templates for adaptation.                                         |
| `--use_avg_embeddings` | `False`     | Use averaged text embeddings for prototypes.                                |
| `--disable_wandb`      | `False`     | Disable Weights & Biases logging.                                           |

---


## Contact

For questions or issues, please open an issue!

Happy experimenting! 🎉

