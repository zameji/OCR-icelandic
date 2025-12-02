import os
import re
import time
from threading import Thread

# Set environment variable FIRST
temp_dir = "/scratch/s5982960/OCR-icelandic/temp"
os.environ["GRADIO_TEMP_DIR"] = temp_dir
os.makedirs(temp_dir, exist_ok=True)

# Then import gradio
import random

import gradio as gr
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    TextIteratorStreamer,
)

# import subprocess
# subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# print("Running on device:", torch.cuda.get_device_name(0))

model_id = "HuggingFaceTB/SmolVLM-Base"
# model_id = "Sigurdur/SmolVLM-Base-ocr-isl-checkpoint-1500"

processor = AutoProcessor.from_pretrained(model_id)

model_id = "./full_idefics3_lora_merged-ocr-isl-with-isl-bacbone/checkpoint-3000"
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # _attn_implementation="flash_attention_2"
).to("cuda")

# load adapter weights if needed
# model.load_adapter(
#     "./SmolVLM-Base-ocr-isl/checkpoint-4000", adapter_name="checkpoint-4000"
# )

# Load test examples from the dataset
print("Loading example images from dataset...")
dataset = load_dataset("Sigurdur/isl_synthetic_ocr", split="test")
random.seed(42)
example_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
example_images = [dataset[i]["image"] for i in example_indices]
print(f"Loaded {len(example_images)} example images")


def model_inference(
    input_dict,
    history,
    decoding_strategy,
    temperature,
    max_new_tokens,
    repetition_penalty,
    top_p,
):
    text = input_dict["text"]
    text = "Extract the text from the image."
    print(input_dict["files"])
    if len(input_dict["files"]) > 1:
        images = [Image.open(image).convert("RGB") for image in input_dict["files"]]
    elif len(input_dict["files"]) == 1:
        images = [Image.open(input_dict["files"][0]).convert("RGB")]
    else:
        images = []

    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")

    if text == "" and images:
        gr.Error("Please input a text query along the image(s).")

    resulting_messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(len(images))]
            + [{"type": "text", "text": text}],
        }
    ]
    prompt = processor.apply_chat_template(
        resulting_messages, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=[images], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    assert decoding_strategy in [
        "Greedy",
        "Top P Sampling",
    ]
    if decoding_strategy == "Greedy":
        generation_args["do_sample"] = False
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    generation_args.update(inputs)
    # Generate
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_args = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    generated_text = ""

    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    yield "..."
    buffer = ""

    for new_text in streamer:
        buffer += new_text
        generated_text_without_prompt = buffer  # [len(ext_buffer):]
        time.sleep(0.01)
        yield buffer


def ocr_interface(
    image, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p
):
    """Wrapper function to adapt the OCR interface to the model_inference function."""
    input_dict = {
        "text": "Extract the text from the image.",
        "files": [image] if image is not None else [],
    }
    history = []

    # Stream the inference results
    for output in model_inference(
        input_dict,
        history,
        decoding_strategy,
        temperature,
        max_new_tokens,
        repetition_penalty,
        top_p,
    ):
        yield output


with gr.Blocks() as demo:
    gr.Markdown("# Icelandic OCR (Checkpoint 200) - SmolVLM")
    gr.Markdown(
        "Upload an image to extract Icelandic text using [SmolVLM-Base-ocr-isl-checkpoint-200](https://huggingface.co/Sigurdur/SmolVLM-Base-ocr-isl-checkpoint-200)."
    )
    gr.Markdown(
        "- Fine-tuned on [isl_synthetic_ocr dataset](https://huggingface.co/datasets/Sigurdur/isl_synthetic_ocr)"
    )
    gr.Markdown(
        "- Follow me on [Linkedin](https://www.linkedin.com/in/sigurdur-haukur-birgisson/) and [Hugging Face](https://huggingface.co/Sigurdur) for more updates!"
    )
    gr.Markdown(
        "Note: This model is fine-tuned specifically for Icelandic OCR tasks and may not perform well on other languages or general VQA tasks. Additionally, the performance might vary for images that don't closely resemble the training data."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")

            with gr.Accordion("Advanced Settings", open=False):
                decoding_strategy = gr.Radio(
                    ["Top P Sampling", "Greedy"],
                    value="Greedy",
                    label="Decoding strategy",
                    info="Higher values is equivalent to sampling more low-probability tokens.",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    value=0.4,
                    step=0.1,
                    interactive=True,
                    label="Sampling temperature",
                    info="Higher values will produce more diverse outputs.",
                )
                max_new_tokens = gr.Slider(
                    minimum=8,
                    maximum=1024,
                    value=512,
                    step=1,
                    interactive=True,
                    label="Maximum number of new tokens to generate",
                )
                repetition_penalty = gr.Slider(
                    minimum=0.01,
                    maximum=5.0,
                    value=1.2,
                    step=0.01,
                    interactive=True,
                    label="Repetition penalty",
                    info="1.0 is equivalent to no penalty",
                )
                top_p = gr.Slider(
                    minimum=0.01,
                    maximum=0.99,
                    value=0.8,
                    step=0.01,
                    interactive=True,
                    label="Top P",
                    info="Higher values is equivalent to sampling more low-probability tokens.",
                )

            with gr.Row():
                cancel_btn = gr.ClearButton(components=[image_input], value="Cancel")
                submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Extracted Text", lines=10, interactive=False
            )

    submit_btn.click(
        fn=ocr_interface,
        inputs=[
            image_input,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=output_text,
        show_progress="full",
    )

    # Add example images
    gr.Markdown("## Example Images from Test Set")
    gr.Examples(
        examples=[[img] for img in example_images],
        inputs=[image_input],
        label="Click an example to load it",
    )

demo.launch(debug=True, share=True)
