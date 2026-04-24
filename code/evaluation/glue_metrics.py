import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


GLUE_METRICS = {
    "cola": ["matthews_correlation"],
    "sst2": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    "qqp": ["accuracy", "f1"],
    "stsb": ["pearson", "spearman"],
    "mnli": ["accuracy"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "wnli": ["accuracy"],
}


def compute_glue_metrics(task, preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    metrics_wanted = GLUE_METRICS[task]

    results = {}
    for m in metrics_wanted:
        if m == "accuracy":
            results["accuracy"] = float(accuracy_score(labels, preds))
        elif m == "f1":
            results["f1"] = float(f1_score(labels, preds))
        elif m == "matthews_correlation":
            results["matthews_correlation"] = float(matthews_corrcoef(labels, preds))
        elif m == "pearson":
            results["pearson"] = float(pearsonr(preds, labels)[0])
        elif m == "spearman":
            results["spearman"] = float(spearmanr(preds, labels)[0])
    return results
