import torch


def _recall_at_k(similarity, k):
    k = min(int(k), similarity.size(1))
    labels = torch.arange(similarity.size(0), device=similarity.device)
    topk = similarity.topk(k, dim=1).indices
    hits = (topk == labels[:, None]).any(dim=1)
    return hits.float().mean().item()


def compute_clip_retrieval_metrics(image_embeds, text_embeds, ks=(1, 5)):
    if image_embeds.size(0) != text_embeds.size(0):
        raise ValueError(
            f"Expected one text per image during eval, got {image_embeds.size(0)} images and {text_embeds.size(0)} texts"
        )

    similarity = image_embeds @ text_embeds.t()
    metrics = {}

    for k in ks:
        metrics[f"image_to_text_R@{k}"] = _recall_at_k(similarity, k)
        metrics[f"text_to_image_R@{k}"] = _recall_at_k(similarity.t(), k)

    if 1 in ks:
        metrics["mean_recall@1"] = 0.5 * (
            metrics["image_to_text_R@1"] + metrics["text_to_image_R@1"]
        )
    if 5 in ks:
        metrics["mean_recall@5"] = 0.5 * (
            metrics["image_to_text_R@5"] + metrics["text_to_image_R@5"]
        )

    return metrics
