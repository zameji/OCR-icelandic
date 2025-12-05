"""
LoRA fine-tuning for the text model within Idefics3.

This script extracts the text model from Idefics3, applies LoRA for efficient fine-tuning,
and prepares the model for training on a custom dataset.

Core idea is to embed icelandic language understanding into the text model, before trying to
fine-tune the full Idefics3 model on image-text pairs.

Before running:

1. activate the uv environment: `source .venv/bin/activate`
2. install requirements: `pip install -r requirements.txt`
3. login to Hugging Face Hub: `huggingface-cli login`
4. configure wandb with `wandb login` then `wandb init` in the root directory of this repo

usage: python train_llm.py push_to_hub=True
"""

import logging
import os
import sys
from dataclasses import dataclass
import transformers
import peft
import time

import torch
from datasets import load_dataset, Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Idefics3ForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    # Dataset configuration
    hf_dataset_id: str = "arnastofnun/IGC-2024"  # Hugging Face dataset ID
    hf_data_directory: str = "wiki"  # Subdirectory within the dataset
    # dataset_split: str = "train[:95%]"  # Dataset split to use for training
    # eval_dataset_split: str = "train[95%:100%]"  # Dataset split to use for evaluation
    max_length: int = 512  # Max token length for text sequences
    max_entries: int = 0  # Max entries to process from dataset (for quick testing, 0 = all)
    max_eval_entries: int = 10  # Max entries to process from eval dataset (for quick testing, 0 = all)
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
    save_total_limit: int = 3  # Limit the total amount of checkpoints. Deletes the older checkpoints.
    load_best_model_at_end: bool = True  # Load best model at end of training
    eval_strategy: str = "steps"  # Evaluation strategy during training
    fp16: bool = True  # Use mixed precision training if True
    dataloader_drop_last: bool = True  # Drop last incomplete batch if True
    remove_unused_columns: bool = False  # Whether to remove unused columns in dataset
    load_best_model_at_end: bool = True  # Load best model at end of training
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


import time

class TokenCountCallback(TrainerCallback):
    def __init__(self, max_length, tokenizer=None):
        self.total_tokens = 0
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.step_start_time = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Calculate tokens per step
        tokens_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.max_length
        self.total_tokens += tokens_per_step

        # Calculate timing
        step_time = time.time() - self.step_start_time if self.step_start_time else 0

        # Log every logging_steps
        if state.global_step % args.logging_steps == 0:
            log_data = {
                "total_tokens_trained": self.total_tokens,
                "tokens_this_step": tokens_per_step,
                "step_time_seconds": step_time,
                "epoch_progress": state.epoch,
            }

            if step_time > 0:
                log_data["tokens_per_second"] = tokens_per_step / step_time

            # Add GPU memory if available
            if torch.cuda.is_available():
                log_data["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1024**3
                log_data["gpu_memory_cached_gb"] = torch.cuda.memory_reserved() / 1024**3

            wandb.log(log_data, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time if self.start_time else 0
        wandb.log({
            "final_total_tokens": self.total_tokens,
            "total_training_time_minutes": total_time / 60,
            "average_tokens_per_second": self.total_tokens / total_time if total_time > 0 else 0,
        })


class EvaluationCallback(TrainerCallback):
    """Custom callback to log additional evaluation metrics"""

    def __init__(self, tokenizer, eval_prompts=None):
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts or [
            "Einu sinni var karl og kerling sem bjuggu í",
            "Ísland er fallegt land þar sem",
            "Reykjavík er höfuðborg",
        ]

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        """Generate sample text during evaluation to monitor quality"""
        if model is None:
            return

        model.eval()
        generated_samples = {}

        for i, prompt in enumerate(self.eval_prompts):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_samples[f"eval_sample_{i}"] = generated_text
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                generated_samples[f"eval_sample_{i}"] = f"Generation failed: {str(e)}"

        # Log to wandb
        wandb.log(generated_samples, step=state.global_step)


# Inference function for text generation
def generate_text(
    text_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 50,
) -> str:
    """Generate text from the model given a prompt."""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = text_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def sanity_check(text_model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Sanity check to see if the model can generate Icelandic text."""
    # Sanity check - generate text before training
    prompt = "Einu sinni var karl og kerling sem bjuggu í"
    logger.info("Before training:")
    logger.info(generate_text(text_model, tokenizer, prompt))


def prepare_datasets(
    cfg: TrainConfig, tokenizer: AutoTokenizer
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and tokenizes both training and evaluation datasets.

    Args:
        cfg (TrainConfig): Configuration for dataset and training.
        tokenizer (AutoTokenizer): Tokenizer for the text model.

    Returns:
        tuple: (train_dataset, eval_dataset) - Tokenized datasets ready for training.
    """

    ds = load_dataset(
        cfg.hf_dataset_id,
        cfg.hf_data_directory,
    )

    logger.info(f"Dataset loaded: {ds}")


    # split and shuffle the dataset
    train_ds, eval_ds = ds["train"].train_test_split(
        test_size=0.2, seed=42
    ).values()

    eval_ds, test_ds = eval_ds.train_test_split(
        test_size=0.5, seed=42
    ).values()

    # aim for 80% train, 10% eval, 10% test split (and at least 2k samples in each)
    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Evaluation dataset size: {len(eval_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")

    # for now we will not use the test set, but it could be used for final evaluation after training

    # Tokenize dataset
    def tokenize_function(examples):
        # Tokenize without truncation or padding to preserve all content
        return tokenizer(examples[cfg.text_key], truncation=False, padding=False)

    logger.info("Tokenizing training dataset...")
    train_dataset = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    logger.info("Tokenizing evaluation dataset...")
    eval_dataset = eval_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    # Add packing step: concatenate all input_ids and split into chunks
    def pack_dataset(dataset):
        # Concatenate all sequences
        all_input_ids = []
        for example in dataset:
            all_input_ids.extend(example["input_ids"])
            all_input_ids.append(tokenizer.eos_token_id)  # Add EOS token between examples

        # Split into chunks of max_length
        packed_input_ids = []
        for i in range(0, len(all_input_ids), cfg.max_length):
            chunk = all_input_ids[i : i + cfg.max_length]
            if len(chunk) == cfg.max_length:  # Only keep full chunks
                packed_input_ids.append(chunk)

        # Convert to dataset format
        return {"input_ids": packed_input_ids}

    train_dataset = pack_dataset(train_dataset)
    eval_dataset = pack_dataset(eval_dataset)

    # Convert back to Dataset objects if needed (using datasets.Dataset.from_dict)
    train_dataset = Dataset.from_dict(train_dataset)
    eval_dataset = Dataset.from_dict(eval_dataset)

    # Limit dataset sizes if specified
    if cfg.max_entries > 0:
        train_dataset = train_dataset.select(range(min(cfg.max_entries, len(train_dataset))))

    if cfg.max_eval_entries > 0:
        eval_dataset = eval_dataset.select(range(min(cfg.max_eval_entries, len(eval_dataset))))

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def get_text_model_from_idefics3(
    model: Idefics3ForConditionalGeneration,
) -> AutoModelForCausalLM:
    """Extracts the text model (Llama) from the Idefics3 model.

    note: This function takes around 1 minute to run on a NVIDIA L40s (48 GB VRAM)

    The key steps are:
    - Load just the text model configuration from Idefics3
    - Create a new AutoModelForCausalLM instance with that config
    - Remap the state dict from Idefics3 to match what AutoModelForCausalLM expects
    - Load the remapped state dict into the new text model

    Args:
        model (Idefics3ForConditionalGeneration): The full Idefics3 model.
    Returns:
        AutoModelForCausalLM: The extracted text model.
    """

    # Load just the text model (Llama) directly
    logger.info("Loading text model from Idefics3...")
    config = model.config.text_config
    text_model = AutoModelForCausalLM.from_config(config)

    # The text_model from Idefics3 doesn't have the outer 'model' wrapper
    # but AutoModelForCausalLM expects it. So we need to adjust the state dict:
    # Get the state dict from Idefics3's text model
    source_state = model.model.text_model.state_dict()

    # Add the 'model.' prefix to match what AutoModelForCausalLM expects
    remapped_state = {}
    for key, value in source_state.items():
        remapped_state[f"model.{key}"] = value

    # Also add the lm_head weights
    remapped_state["lm_head.weight"] = model.lm_head.weight

    # Now load everything at once
    logger.info("Loading state dict into text model...")
    text_model.load_state_dict(remapped_state, strict=True)

    return text_model

def lora_merge_and_save_full_model(model, text_model, tokenizer, cfg):
   # test lora merging and saving full model
    lora_merged = text_model.merge_and_unload()

    # Test the LoRA model BEFORE merging
    test_input = tokenizer("Einu sinni var karl og kerling sem bjuggu í", return_tensors="pt").input_ids.to(DEVICE)
    lora_output = text_model.generate(test_input, max_new_tokens=10, do_sample=False)
    lora_decoded = tokenizer.decode(lora_output[0], skip_special_tokens=True)
    print(f"LoRA text model: {lora_decoded}")

    # Get the merged state dict and map it back to the original text model
    merged_state_dict = lora_merged.state_dict()

    text_model_state_dict = {}
    lm_head_weight = None
    for key, value in merged_state_dict.items():
        if key == 'lm_head.weight':
            lm_head_weight = value  # Store for later use
        elif key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            text_model_state_dict[new_key] = value
        else:
            text_model_state_dict[key] = value

    # Load the weights back into the original text model structure
    model.model.text_model.load_state_dict(text_model_state_dict)

    if lm_head_weight is not None:
        model.lm_head.weight.data = lm_head_weight.data


    # save the full model with LoRA merged
    model.save_pretrained("full_idefics3_lora_merged")

    # load the merged model to verify it works
    loaded_model = Idefics3ForConditionalGeneration.from_pretrained(
        "full_idefics3_lora_merged",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(DEVICE)

    # After loading the saved model, test it
    loaded_output = loaded_model.generate(test_input, max_new_tokens=10, do_sample=False)
    loaded_decoded = tokenizer.decode(loaded_output[0], skip_special_tokens=True)
    print(f"Loaded model: {loaded_decoded}")

        # Optionally push to Hugging Face Hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info(f"Pushing model to the hub at {cfg.hub_repo_id}...")
        model.push_to_hub(cfg.hub_repo_id)


def fine_tune_text_model(cfg: TrainConfig) -> None:
    """
    Main function to fine-tune the text model within Idefics3 using LoRA.
    Args:
        cfg (TrainConfig): Configuration for dataset, model, and training.
    Returns:
        None
    """

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Loading model {cfg.model_id}...")

    # Load the original model
    model = Idefics3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    ).to(DEVICE)

    logger.info("Model loaded.")

    logger.info("Extracting text model from Idefics3...")
    # Extract the text model (Llama)
    text_model = get_text_model_from_idefics3(model)

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        cfg.model_id
    )
    tokenizer = processor.tokenizer

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_r,  # Rank of adaptation - higher values = more parameters but potentially better performance
        lora_alpha=cfg.lora_alpha,  # LoRA scaling parameter - typically 2x the rank
        lora_dropout=cfg.lora_dropout,  # Dropout for LoRA layers
        target_modules=[
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            "gate_proj",  # Gate projection (for feedforward)
            "up_proj",  # Up projection (for feedforward)
            "down_proj",  # Down projection (for feedforward)
        ],
        bias="none",  # Whether to train bias parameters
        use_rslora=False,  # Use rank-stabilized LoRA
    )

    # Apply LoRA to the model
    text_model = get_peft_model(text_model, lora_config)

    # the base_model_name_or_path key in the config causes issues when loading later, so we need to update it
    text_model.peft_config['default'].base_model_name_or_path = cfg.model_id

    # Move to device
    text_model.to(DEVICE)

    # Print trainable parameters info
    text_model.print_trainable_parameters()

    # sanity check before training
    sanity_check(text_model, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # False for causal LM (GPT-style), True for masked LM (BERT-style)
    )

    # Prepare both training and evaluation datasets
    train_dataset, eval_dataset = prepare_datasets(cfg, tokenizer)

    # Initialize wandb
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=cfg.entity,
        # Set the wandb project where this run will be logged.
        project=cfg.project,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.per_device_train_batch_size,
            "eval_batch_size": cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_train_epochs": cfg.num_train_epochs,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "model_id": cfg.model_id,
            "hf_dataset_id": cfg.hf_dataset_id,
            "hf_data_directory": cfg.hf_data_directory,
            # "dataset_split": cfg.dataset_split,
            # "eval_dataset_split": cfg.eval_dataset_split,
            "max_length": cfg.max_length,
            "max_entries": cfg.max_entries,
            "max_eval_entries": cfg.max_eval_entries,
            "push_to_hub": cfg.push_to_hub,
            "hub_repo_id": cfg.hub_repo_id,
            "fp16": cfg.fp16,
            "eval_steps": cfg.eval_steps,
            "eval_strategy": cfg.eval_strategy,
            # Hardware info
            "device": DEVICE,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",

            # Dataset info
            "total_train_dataset_size": len(train_dataset),
            "total_eval_dataset_size": len(eval_dataset),
            "effective_batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            "total_training_steps": len(train_dataset) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps) * cfg.num_train_epochs,

            # Model architecture
            "model_size": sum(p.numel() for p in text_model.parameters()),
            "trainable_params": sum(p.numel() for p in text_model.parameters() if p.requires_grad),
            "lora_target_modules": lora_config.target_modules,

            # Environment
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        },
    )

    # Training arguments - adjusted for LoRA with evaluation
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        dataloader_drop_last=cfg.dataloader_drop_last,
        remove_unused_columns=cfg.remove_unused_columns,
        report_to=cfg.report_to,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
    )

    # Create callbacks
    token_callback = TokenCountCallback(max_length=cfg.max_length, tokenizer=tokenizer)
    eval_callback = EvaluationCallback(tokenizer=tokenizer)

    # Create trainer with evaluation dataset
    trainer = Trainer(
        model=text_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add evaluation dataset
        data_collator=data_collator,
        callbacks=[token_callback, eval_callback],
    )

    # Train the model
    logger.info("Starting LoRA training...")
    trainer.train()

    # Save the LoRA adapters
    logger.info("Saving LoRA adapters...")
    text_model.save_pretrained(cfg.output_dir)

    # Log final evaluation metrics
    logger.info("Running final evaluation...")
    final_eval_results = trainer.evaluate()
    wandb.log({"final_eval_loss": final_eval_results["eval_loss"]})

    # merge and save the full model with LoRA weights
    lora_merge_and_save_full_model(model, text_model, tokenizer, cfg)


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    fine_tune_text_model(cfg)


if __name__ == "__main__":
    main()