from datasets import load_dataset
from PIL import Image


def _is_image_feature(feature):
    return feature.__class__.__name__.lower() == "image"


def _normalize_text(text):
    if isinstance(text, list):
        if not text:
            return ""
        text = text[0]
    return str(text)


def find_image_text_columns(dataset):
    image_col = None
    text_col = None

    for name, feature in dataset.features.items():
        if image_col is None and _is_image_feature(feature):
            image_col = name

    text_candidates = ["caption", "text", "sentence", "description", "question", "prompt"]
    for candidate in text_candidates:
        for name in dataset.column_names:
            if candidate in name.lower():
                text_col = name
                break
        if text_col is not None:
            break

    if image_col is None:
        raise ValueError(f"Could not infer image column from {dataset.column_names}")
    if text_col is None:
        raise ValueError(f"Could not infer text column from {dataset.column_names}")

    return image_col, text_col


def load_clip_retrieval_data(dataset_name, split="train", max_samples=None, eval_ratio=0.2, seed=42):
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)

    if max_samples is not None:
        max_samples = min(int(max_samples), len(dataset))
        dataset = dataset.select(range(max_samples))

    split_ds = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    image_col, text_col = find_image_text_columns(split_ds["train"])

    return {
        "train": split_ds["train"],
        "validation": split_ds["test"],
        "image_col": image_col,
        "text_col": text_col,
    }


def make_clip_collate_fn(processor, image_col, text_col, max_length=None):
    def collate_fn(batch):
        images = []
        texts = []

        for example in batch:
            image = example[image_col]
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            images.append(image.convert("RGB"))
            texts.append(_normalize_text(example[text_col]))

        return processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    return collate_fn
