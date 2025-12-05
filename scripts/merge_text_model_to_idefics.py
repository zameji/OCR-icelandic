"""
Swap the language model component in Idefics3 with our fine-tuned model.

This script takes our already-merged fine-tuned text model and replaces
the language model component in the original Idefics3, keeping all the
vision and multimodal components intact.
"""

import logging

import torch
from transformers import AutoModel, AutoTokenizer, Idefics3ForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def swap_language_model_in_idefics3(
    original_idefics_model_id: str,
    fine_tuned_text_model_id: str,
    output_path: str = "./idefics3_with_icelandic_lm",
    push_to_hub: bool = False,
    hub_repo_id: str = None,
) -> Idefics3ForConditionalGeneration:
    """
    Replace the language model in Idefics3 with our fine-tuned model.

    Args:
        original_idefics_model_id (str): Original Idefics3 model ID
        fine_tuned_text_model_path (str): Path to our merged fine-tuned text model
        output_path (str): Path to save the modified Idefics3 model
        push_to_hub (bool): Whether to push to Hugging Face Hub
        hub_repo_id (str): Hub repository ID for pushing

    Returns:
        Idefics3ForConditionalGeneration: The modified Idefics3 model
    """

    logger.info(f"Loading original Idefics3 model: {original_idefics_model_id}")

    # Load the original Idefics3 model (this has the vision components we want to keep)
    idefics_model = Idefics3ForConditionalGeneration.from_pretrained(
        original_idefics_model_id,
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"Loading our fine-tuned text model: {fine_tuned_text_model_id}")

    # Load our fine-tuned text model
    fine_tuned_model = AutoModel.from_pretrained(
        fine_tuned_text_model_id,
        torch_dtype=torch.bfloat16,
    )

    logger.info("Swapping language model components...")

    # Get the state dict from your fine-tuned model
    fine_tuned_state = fine_tuned_model.state_dict()

    # Now we need to map the weights from our fine-tuned model back into Idefics3

    for key, value in fine_tuned_state.items():
        if key.startswith("model."):
            # Remove the 'model.' prefix to match Idefics3's text_model structure
            text_model_key = key[6:]  # Remove 'model.' (6 characters)

            # Check if this parameter exists in the Idefics3 text model
            if text_model_key in idefics_model.model.text_model.state_dict():
                logger.debug(f"Copying {text_model_key}")
                # Copy the parameter directly
                idefics_model.model.text_model.state_dict()[text_model_key].copy_(value)
            else:
                logger.warning(f"Key {text_model_key} not found in Idefics3 text model")

        elif key == "lm_head.weight":
            # Update the language model head
            logger.debug("Copying lm_head.weight")
            idefics_model.lm_head.weight.data.copy_(value)

    logger.info("Language model swap complete!")

    # Load tokenizer (use the one from your fine-tuned model if it's different)
    try:
        # Try to load tokenizer from your fine-tuned model first
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_text_model_id)
        logger.info("Using tokenizer from fine-tuned model")
    except:
        # Fall back to original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(original_idefics_model_id)
        logger.info("Using tokenizer from original Idefics3 model")

    logger.info(f"Saving modified Idefics3 model to: {output_path}")

    # Save the modified model
    idefics_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Optionally push to hub
    if push_to_hub and hub_repo_id:
        logger.info(f"Pushing model to: {hub_repo_id}")
        idefics_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)

    return idefics_model


def test_swapped_model(model_path: str) -> None:
    """
    Test the model with swapped language component.

    Args:
        model_path (str): Path to the modified model
    """
    logger.info("Testing swapped model...")

    # Load the modified model
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test Icelandic text generation
    prompt = "Einu sinni var karl og kerling sem bjuggu í"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    logger.info(f"Testing with prompt: '{prompt}'")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")

    # Test that the model still loads properly as Idefics3
    logger.info("Verifying model structure...")
    logger.info(f"Model type: {type(model)}")
    logger.info(
        f"Has vision model: {hasattr(model, 'vision_model') or hasattr(model.model, 'vision_model')}"
    )
    logger.info(f"Has text model: {hasattr(model.model, 'text_model')}")
    logger.info(f"Has lm_head: {hasattr(model, 'lm_head')}")

    logger.info("Test complete - model structure looks good!")


def compare_before_after(
    original_model_id: str,
    modified_model_path: str,
    test_prompt: str = "Einu sinni var karl og kerling sem bjuggu í",
) -> None:
    """
    Compare text generation before and after the swap.

    Args:
        original_model_id (str): Original Idefics3 model
        modified_model_path (str): Path to modified model
        test_prompt (str): Test prompt for comparison
    """
    logger.info("Comparing original vs modified model...")

    # Load original model
    logger.info("Loading original model...")
    original_model = Idefics3ForConditionalGeneration.from_pretrained(
        original_model_id,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_id)

    # Load modified model
    logger.info("Loading modified model...")
    modified_model = Idefics3ForConditionalGeneration.from_pretrained(
        modified_model_path,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    modified_tokenizer = AutoTokenizer.from_pretrained(modified_model_path)

    # Test both models
    inputs_original = original_tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    inputs_modified = modified_tokenizer(test_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Original model
        outputs_original = original_model.generate(
            **inputs_original,
            max_length=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )

        # Modified model
        outputs_modified = modified_model.generate(
            **inputs_modified,
            max_length=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )

    original_text = original_tokenizer.decode(
        outputs_original[0], skip_special_tokens=True
    )
    modified_text = modified_tokenizer.decode(
        outputs_modified[0], skip_special_tokens=True
    )

    logger.info(f"Prompt: '{test_prompt}'")
    logger.info(f"Original model output: {original_text}")
    logger.info(f"Modified model output: {modified_text}")


def main():
    """Main function to swap the language models"""

    # Configuration
    original_idefics_model_id = (
        "HuggingFaceTB/SmolVLM-Base"  # Our original Idefics3 model
    )
    fine_tuned_text_model_path = (
        "Sigurdur/SmolVLM-Base-ICELANDIC"  # Path to our merged fine-tuned text model
    )
    output_path = "./idefics3_with_icelandic"  # Where to save the modified Idefics3

    # Optional: push to hub
    push_to_hub = False
    hub_repo_id = "Sigurdur/SmolVLM-Icelandic"  # Your hub repo

    # Perform the swap
    modified_model = swap_language_model_in_idefics3(
        original_idefics_model_id=original_idefics_model_id,
        fine_tuned_text_model_id=fine_tuned_text_model_path,
        output_path=output_path,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
    )

    # Test the modified model
    test_swapped_model(output_path)

    # Compare before and after
    compare_before_after(
        original_idefics_model_id,
        output_path,
        "Einu sinni var karl og kerling sem bjuggu í",
    )

    logger.info("All done! Our Idefics3 model now has Icelandic language capabilities.")


if __name__ == "__main__":
    main()
