from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Configuration dataclass for training SmolVLM with LoRA on Icelandic text data.
    """

    # Dataset configuration
    hf_dataset_id: str = "arnastofnun/IGC-2024"  # Hugging Face dataset ID
    hf_data_directory: str = "wiki"  # Subdirectory within the dataset
    # dataset_split: str = "train[:95%]"  # Dataset split to use for training
    # eval_dataset_split: str = "train[95%:100%]"  # Dataset split to use for evaluation
    max_length: int = 512  # Max token length for text sequences
    max_entries: int = (
        0  # Max entries to process from dataset (for quick testing, 0 = all)
    )
    max_eval_entries: int = (
        10  # Max entries to process from eval dataset (for quick testing, 0 = all)
    )
    text_key: str = "document"  # Key in dataset containing text data

    # Model configuration
    model_id: str = "HuggingFaceTB/SmolVLM-Base"  # Base model ID
    push_to_hub: bool = False  # Whether to push trained model to Hugging Face Hub
    hub_repo_id: str = (
        "Sigurdur/SmolVLM-Base-ICELANDIC"  # Hugging Face repo ID to push model
    )

    # LoRA configuration
    lora_r: int = 32  # Rank of adaptation - higher values = more parameters but potentially better performance
    lora_alpha: int = 64  # LoRA scaling parameter - typically 2x the rank
    lora_dropout: float = 0.1  # Dropout for LoRA layers

    # Training arguments
    output_dir: str = (
        "./lora_results"  # Directory to save LoRA adapters and checkpoints
    )
    per_device_train_batch_size: int = 8  # Batch size per device during training
    per_device_eval_batch_size: int = 8  # Batch size per device during evaluation
    gradient_accumulation_steps: int = 4  # Number of steps to accumulate gradients
    num_train_epochs: int = 1  # Total number of training epochs
    learning_rate: float = 1e-4  # Learning rate for optimizer
    warmup_steps: int = 100  # Number of warmup steps for learning rate scheduler
    logging_steps: int = 50  # Log every X updates steps
    eval_steps: int = 200  # Evaluate every X steps
    save_strategy: str = "steps"  # Save checkpoint every X steps
    save_steps: int = 200  # Save checkpoint every X steps
    save_total_limit: int = (
        3  # Limit the total amount of checkpoints. Deletes the older checkpoints.
    )
    load_best_model_at_end: bool = True  # Load best model at end of training
    eval_strategy: str = "steps"  # Evaluation strategy during training
    fp16: bool = True  # Use mixed precision training if True
    dataloader_drop_last: bool = True  # Drop last incomplete batch if True
    remove_unused_columns: bool = False  # Whether to remove unused columns in dataset
    metric_for_best_model: str = "eval_loss"  # Metric to use for best model selection
    greater_is_better: bool = False  # Whether greater metric is better (False for loss)

    # logging configuration
    report_to: str = "wandb"  # Report to wandb
    entity: str = "sigurdurhaukur-team"  # wandb entity
    project: str = "smolVLM"  # wandb project name
    run_name: str = "lora-finetune-icelandic"  # wandb run name
    run_description: str = "LoRA fine-tuning of SmolVLM text model on Icelandic text data"  # wandb run description
    tags: list = (
        "LoRA",
        "Idefics3",
        "SmolVLM",
        "Icelandic",
        "Fine-tuning",
        "NLP",
        "Vision-Language Model",
    )  # wandb tags
