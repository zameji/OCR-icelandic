import logging
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from Levenshtein import distance as lev_distance
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

import wandb

USE_LORA = True
USE_QLORA = False
SMOL = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"

processor = AutoProcessor.from_pretrained(model_id) # use the processor from the base model

# our custom model, with pre-trained LLM backbone on Icelandic text
model_id = "./full_idefics3_lora_merged"


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


ds = load_dataset("Sigurdur/isl_synthetic_ocr", trust_remote_code=True)

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
wandb.init(
    project="SmolVLM",
    config={
        "model_name": model_id.split("/")[-1],
        "dataset": "Sigurdur/isl_synthetic_ocr",
        "use_lora": USE_LORA,
        "use_qlora": USE_QLORA,
        "epochs": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
    },
)

training_args = TrainingArguments(
    output_dir=f"./{model_id.split('/')[-1]}-ocr-isl-with-isl-bacbone",
    hub_model_id=f"{model_id.split('/')[-1]}-ocr-isl-with-isl-backbone",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=125,
    optim="paged_adamw_8bit",
    bf16=True,
    report_to="wandb",
    remove_unused_columns=False,
    gradient_checkpointing=True,
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
