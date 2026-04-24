from transformers import RobertaForSequenceClassification

from lora.inject import inject_lora
from lora.targets import get_target_modules

def build_roberta_lora(model_name, num_labels, rank, alpha, dropout=0.1):
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    targets = get_target_modules("roberta")
    model, replaced = inject_lora(model, targets, rank=rank, alpha=alpha, dropout=dropout)
    # Unfreeze the classification head — it's randomly initialized, needs training
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model, replaced