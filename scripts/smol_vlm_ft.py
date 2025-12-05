import logging
import sys
from typing import Optional

import numpy as np
import peft
import torch
import transformers
from datasets import load_dataset
from helpers import TrainConfig
from Levenshtein import distance as lev_distance
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fintune_smolvlm_ocr(cfg: TrainConfig) -> None:
    """Fine-tune SmolVLM on Icelandic OCR dataset."""
    USE_LORA = True
    USE_QLORA = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model_id
        )  # use the processor from the base model

    except Exception as e:
        logger.error(f"Error loading processor for model {cfg.model_id}: {e}")
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
        logger.info("Loaded default processor HuggingFaceTB/SmolVLM-Base")

    # our custom model, with pre-trained LLM backbone on Icelandic text
    model_id = cfg.model_id  # ./full_idefics3_lora_merged
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=[
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
            ],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian",
        )
        lora_config.inference_mode = False
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print("Loading model...")
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            # _attn_implementation="flash_attention_2",
            device_map="auto",
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2",
        ).to(DEVICE)

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    ds = load_dataset(cfg.hf_dataset_id, trust_remote_code=True)

    train_ds = ds["train"]
    # validation_ds = ds["validation"]
    validation_ds = ds["validation"].select(range(min(5, len(ds["validation"]))))

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            task_desc = "Extract the text from the image."
            answer = example["text"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": task_desc},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    def compute_text_metrics(predictions, labels):
        """Compute OCR metrics using Levenshtein distance library."""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Handle logits
        if predictions.ndim > 2:
            predictions = np.argmax(predictions, axis=-1)

        # Decode predictions and labels
        label_ids = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Special Icelandic characters
        special_chars = set("þðáéíóúýæö")

        # Initialize metrics
        metrics = {
            "total_words": 0,
            "word_errors": 0,
            "total_chars": 0,
            "char_errors": 0,
            "exact_matches": 0,
            "special_correct": 0,
            "special_total": 0,
            "seq_acc_5": 0,
            "seq_acc_10": 0,
        }

        for pred, label in zip(decoded_preds, decoded_labels):
            # Exact match
            if pred == label:
                metrics["exact_matches"] += 1

            # Word-level metrics (using Levenshtein)
            pred_words = pred.split()
            label_words = label.split()
            metrics["word_errors"] += lev_distance(pred_words, label_words)
            metrics["total_words"] += len(label_words)

            # Character-level metrics
            char_dist = lev_distance(pred, label)
            metrics["char_errors"] += char_dist
            metrics["total_chars"] += len(label)

            # Sequence accuracy thresholds
            if len(label) > 0:
                cer = char_dist / len(label)
                if cer < 0.05:
                    metrics["seq_acc_5"] += 1
                if cer < 0.10:
                    metrics["seq_acc_10"] += 1

            # Special character accuracy (position-based): Old method, but its too strict
            # for i, char in enumerate(label.lower()):
            #     if char in special_chars:
            #         metrics["special_total"] += 1
            #         if i < len(pred) and pred.lower()[i] == char:
            #             metrics["special_correct"] += 1

            # Special character accuracy (count-based)
            # Since we also report CER, this is acceptable
            for char in special_chars:
                label_count = label.lower().count(char)
                pred_count = pred.lower().count(char)
                metrics["special_total"] += label_count
                metrics["special_correct"] += min(label_count, pred_count)

        n = len(decoded_labels)

        return {
            "wer": metrics["word_errors"] / max(metrics["total_words"], 1),
            "cer": metrics["char_errors"] / max(metrics["total_chars"], 1),
            "exact_match": metrics["exact_matches"] / n,
            "special_char_acc": metrics["special_correct"]
            / max(metrics["special_total"], 1),
            "seq_acc_5": metrics["seq_acc_5"] / n,
            "seq_acc_10": metrics["seq_acc_10"] / n,
        }

    def compute_metrics(eval_preds):
        """Compute metrics function for Trainer."""

        # eval_preds is an EvalPrediction object with predictions and label_ids attributes
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

        # Handle tuple predictions (e.g., when model returns logits and hidden states)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        return compute_text_metrics(predictions, labels)

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
            "gpu_name": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "CPU",
            # Dataset info
            "total_train_dataset_size": len(train_ds),
            "total_eval_dataset_size": len(validation_ds),
            "effective_batch_size": cfg.per_device_train_batch_size
            * cfg.gradient_accumulation_steps,
            "total_training_steps": len(train_ds)
            // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
            * cfg.num_train_epochs,
            # Model architecture
            "model_size": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "lora_target_modules": lora_config.target_modules,
            # Environment
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        },
    )

    training_args = TrainingArguments(
        output_dir=f"./{model_id.split('/')[-1]}-ocr-isl-with-isl-bacbone",
        hub_model_id=f"{model_id.split('/')[-1]}-ocr-isl-with-isl-backbone",
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        optim="paged_adamw_8bit",
        bf16=cfg.fp16,
        report_to=cfg.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()


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

    fintune_smolvlm_ocr(cfg)


if __name__ == "__main__":
    main()
