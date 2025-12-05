# OCR-icelandic: Language-Agnostic OCR Model Training Pipeline

A comprehensive training recipe for creating optical character recognition (OCR) Vision Transformer models for languages with limited or no existing image-text paired datasets. This project demonstrates how to bootstrap OCR capabilities for any language given only a text corpus.

## Overview

Training OCR models traditionally requires large datasets of document images paired with their transcriptions. This is a significant bottleneck for low-resource and underrepresented languages. **OCR-icelandic** solves this problem by:

1. **Synthetic Data Generation**: Creates realistic document images from plain text by rendering text with various fonts, sizes, colors, and transformations
2. **Model Fine-tuning**: Fine-tunes vision-language models (like SmolVLM) using LoRA adapters on the synthetic data
3. **Language Customization**: Provides a flexible pipeline that works for any language with an available text corpus

This project was initially developed for Icelandic but is designed to be language-agnostic and easily adaptable to any language.

## Key Features

- **Synthetic OCR Dataset Generation**: Transform plain text into visually diverse document images
- **Vision Transformer Fine-tuning**: Fine-tune models like SmolVLM and IDEFICS using efficient LoRA adapters
- **Multi-language Support**: Works with any language with a text dataset
- **Minimal Requirements**: Only needs a text corpus in the target language
- **Production-Ready Pipeline**: SLURM integration for cluster computing
- **Web UI for Inference**: Interactive interface for testing trained models

## Project Structure

```
├── scripts/                          # Core training and processing scripts
│   ├── prepare_data.py              # Generates synthetic OCR dataset from text
│   ├── smol_vlm_ft.py               # Fine-tunes SmolVLM with LoRA
│   ├── train_llm.py                 # Text-to-text model fine-tuning
│   ├── merge_text_model_to_idefics.py  # Model merging utilities
│   ├── build_gold_data.py           # Builds evaluation datasets
│   ├── helpers.py                   # Configuration dataclasses
│   └── webui.py                     # Gradio-based inference interface
├── src/ocr_icelandic/               # Core library modules
│   └── utils.py                     # Image generation and transformation utilities
├── notebooks/                        # Jupyter notebooks for experimentation
│   ├── smol_vlm_inference.ipynb     # Inference demonstrations
│   └── Smol_VLM_FT.ipynb            # Fine-tuning walkthrough
├── slurm/                           # SLURM job submission scripts
│   ├── generate_synthetic_data.slurm
│   ├── train_smolVLM.slurm
│   └── train_smolVLM_LLM.slurm
├── models/                          # Trained model checkpoints
│   ├── SmolVLM-Base-ocr-isl/        # LoRA adapters
│   ├── full_idefics3_lora_merged/   # Merged model weights
│   └── lora_results/                # Training outputs
├── isl_synthetic_ocr_output/        # Generated synthetic datasets
└── pyproject.toml                   # Project dependencies
```

## Workflow

### Step 1: Prepare Text Data

Start with a text dataset in your target language. The project supports loading from:
- Hugging Face Datasets (e.g., `arnastofnun/IGC-2024` for Icelandic)
- Local text files
- Custom data sources

### Step 2: Generate Synthetic OCR Dataset

Use `prepare_data.py` to render text as document images with realistic variations:

```bash
python scripts/prepare_data.py
```

This generates:
- Synthetic document images with text rendered in various fonts and sizes
- Random visual transformations (rotations, skewing, noise)
- Train/validation/test splits
- HuggingFace-compatible dataset format

**Configuration** (via `DataConfig` in `helpers.py`):
- `dataset_path`: Source text dataset
- `text_column`: Which column contains the text
- `max_entries`: Number of examples to generate
- `image_width/height`: Document image dimensions
- `font_path`: Path to font file supporting the target language
- `font_size`: Text rendering size
- `output_path`: Where to save the dataset

### Step 3: Fine-tune Vision Transformer Model

Fine-tune a vision-language model on your synthetic dataset using LoRA:

```bash
python scripts/smol_vlm_ft.py
```

**Supported Models**:
- SmolVLM (recommended for efficiency)
- IDEFICS3
- Other HuggingFace vision-language models

**Training Configuration** (via `TrainConfig`):
- `lora_r`: LoRA rank (controls adapter capacity)
- `lora_alpha`: LoRA scaling factor
- `per_device_train_batch_size`: Batch size
- `num_train_epochs`: Training duration
- `learning_rate`: Optimizer learning rate

### Step 4: Test & Deploy

Use the fine-tuned model for OCR inference:

```bash
python scripts/webui.py
```

This launches an interactive web interface where you can:
- Upload document images
- See OCR predictions
- Compare with ground truth
- Evaluate model performance

## Installation

### Prerequisites
- Python 3.13+
- CUDA-capable GPU (recommended)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sigurdurhaukur/OCR-icelandic.git
cd OCR-icelandic
```

2. Install dependencies:
```bash
pip install -e .
```

Or with dev dependencies:
```bash
pip install -e ".[dev]"
```

### Dependencies

- **Deep Learning**: `torch`, `transformers`, `peft`, `accelerate`
- **Data Processing**: `datasets`, `pillow`
- **Training Utilities**: `wandb` (experiment tracking), `gradio` (web UI)
- **OCR Specific**: `pyfonts`, `python-levenshtein`
- **Configuration**: `omegaconf`

See `pyproject.toml` for the complete list.

## Quick Start

### For Icelandic OCR

```bash
# 1. Generate synthetic dataset from Icelandic Wikipedia
python scripts/prepare_data.py \
  --dataset_path arnastofnun/IGC-2024 \
  --text_column document \
  --data_directory wiki \
  --max_entries 1000 \
  --output_path isl_synthetic_ocr_output

# 2. Fine-tune SmolVLM on the synthetic data
python scripts/smol_vlm_ft.py \
  --output_dir ./lora_results \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8

# 3. Launch inference UI
python scripts/webui.py
```

### For Other Languages

To adapt this pipeline for your language:

1. **Replace the text dataset**: Update `DataConfig.dataset_path` to point to your language's text corpus
2. **Choose appropriate font**: Ensure the font supports your language's characters (update `DataConfig.font_path`)
3. **Generate synthetic data**: Run `prepare_data.py` with your configuration
4. **Fine-tune the model**: Run `smol_vlm_ft.py` on your synthetic dataset
5. **Evaluate**: Use the web UI to test the trained model

## Advanced Features

### Model Merging

Merge LoRA adapters with base model weights for deployment:

```bash
python scripts/merge_text_model_to_idefics.py
```

### Text-to-Text Model Training

Train an auxiliary LLM for improved text correction:

```bash
python scripts/train_llm.py
```

### SLURM Job Submission

For cluster environments, submit training jobs:

```bash
sbatch slurm/generate_synthetic_data.slurm
sbatch slurm/train_smolVLM.slurm
```

## Model Zoo

Trained models for supported languages:

| Language | Model | Base Model | Status |
|----------|-------|-----------|--------|
| Icelandic | SmolVLM-Base-ocr-isl | SmolVLM-Base | ✅ Available |
| Icelandic | IDEFICS3-ocr-isl | IDEFICS3 | ✅ Available |

## Notebooks

- **`notebooks/Smol_VLM_FT.ipynb`**: Step-by-step fine-tuning tutorial
- **`notebooks/smol_vlm_inference.ipynb`**: Model inference and evaluation examples
- **`notebooks/documentlayoutsynthesis.ipynb`**: Advanced document generation techniques

## Experiment Tracking

Training runs are automatically logged to Weights & Biases. To use it:

1. Install wandb: `pip install wandb`
2. Authenticate: `wandb login`
3. Training logs appear at `https://wandb.ai/your-username/isl-synthetic-ocr`

<!-- ## Performance Notes

- **Training Time**: ~2-4 hours on a single A100 GPU for 1000 synthetic examples
- **Model Size**: SmolVLM LoRA adapters are typically <100MB
- **Inference Speed**: ~100-200ms per image on GPU
- **Accuracy**: Character error rate (CER) typically 5-15% depending on language complexity -->

## Troubleshooting

**Font rendering issues**: Ensure the font file supports all characters in your target language.

**Out of memory**: Reduce `per_device_train_batch_size` or `max_length` in configuration.

**Poor OCR accuracy**: Increase synthetic data diversity or training epochs; verify font quality.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ocr_icelandic_2024,
  title={OCR-icelandic: Language-Agnostic OCR Model Training Pipeline},
  author={Sigurdur Haukur Birgisson},
  year={2025},
  url={https://github.com/sigurdurhaukur/OCR-icelandic}
}
```

## License

[Apache 2.0](./LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

---

**Note**: This project is designed as a reproducible recipe for training OCR models with minimal resources. The approach of using synthetic data generation from text corpora makes it accessible to researchers and practitioners working with any language, regardless of existing OCR dataset availability.