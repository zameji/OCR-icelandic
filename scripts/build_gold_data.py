"""
Writes the metadata.jsonl for huggingface load_dataset to work

Example usage:

my_dataset_repository/
└── images
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
└── text
    ├── 1.txt
    ├── 2.txt
    ├── 3.txt
    ├── 4.txt

Targtet structure:
my_dataset_repository/
└── train
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
    └── metadata.jsonl


Contents of metadata.jsonl:
{"file_name": "1.jpg","text": "a drawing of a green pokemon with red eyes"}
{"file_name": "2.jpg","text": "a green and yellow toy with a red nose"}
{"file_name": "3.jpg","text": "a red and white ball with an angry look on its face"}
{"file_name": "4.jpg","text": "a cartoon ball with a smile on its face"}

usage:
python3 build_gold_data.py image_dir=<path_to_images> text_dir=<path_to_texts> output_dir=<path_to_output> push_to_hub=<True/False>
"""

from omegaconf import OmegaConf
from dataclasses import dataclass
import pathlib as pl
import json
import logging
import sys
from datasets import load_dataset, Dataset, Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class BuildConfig:
    image_dir: str = "images"
    text_dir: str = "text"
    output_dir: str = "my_dataset_repository/train"
    push_to_hub: bool = False
    repo_name: str = "Sigurdur/OCR-Icelandic-benchmark"




def fooberino(cfg: BuildConfig) -> None:
    """does the thing"""

    # get all text files
    image_dir = pl.Path(cfg.image_dir)
    text_dir = pl.Path(cfg.text_dir)
    output_dir = pl.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(image_dir.glob("*.jpg")))
    text_files = sorted(list(text_dir.glob("*.txt")))

    assert len(image_files) == len(text_files), "Number of images and text files must be the same"

    metadata = []
    for img_path, txt_path in zip(image_files, text_files):
        with open(txt_path, "r") as f:
            text = f.read().strip()
        metadata.append({"file_name": img_path.name, "text": text})
        # copy image to output_dir
        dest_path = output_dir / img_path.name
        if not dest_path.exists():
            dest_path.write_bytes(img_path.read_bytes())
            logger.info(f"Copied {img_path} to {dest_path}")
        else:
            logger.info(f"{dest_path} already exists, skipping copy.")

    # write metadata to output_dir/metadata.jsonl
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote metadata to {metadata_path}")

    if cfg.push_to_hub:
        logger.info(f"Pushing {cfg.repo_name} to Hugging Face Hub...")

        # load the dataset with datasets library
        dataset = load_dataset("json", data_files=str(metadata_path))

        # cast the 'file_name' column to 'Image' type and rename to 'image'
        def load_image(example):
            example["image"] = str(output_dir / example["file_name"])
            return example

        dataset = dataset.map(load_image)

        dataset = dataset.cast_column("image", Image())
        dataset = dataset.remove_columns("file_name")

        dataset.push_to_hub(cfg.repo_name)

def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(BuildConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = BuildConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    fooberino(cfg)

if __name__ == "__main__":
    main()

