"""
story_eval.py

This module contains functions to calculate various metrics for evaluating story-to-dialogue conversion tasks.
"""

import evaluate
import pandas as pd
from transformers import AutoTokenizer, EvalPrediction
from config.dir import PREDICTIONS_DIR

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

def calculate_jaccard_similarity_for_texts(text1: str, text2: str) -> float:
    """
    Calculates the Jaccard similarity between two texts.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The Jaccard similarity score.
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def compute_metrics_for_stories(references: list, predictions: list, metrics_prefix: str = "") -> dict:
    """
    Computes various evaluation metrics for story-to-dialogue conversion.

    Args:
        references (list): List of reference dialogues.
        predictions (list): List of predicted dialogues.

    Returns:
        dict: A dictionary containing the computed metrics.
        - rouge1 (float): ROUGE-1 score.
        - rouge2 (float): ROUGE-2 score.
        - rougeL (float): ROUGE-L score.
        - rougeLsum (float): ROUGE-Lsum score.
        - bleu (float): BLEU score.
        - bleu1 (float): BLEU-1 score.
        - bleu2 (float): BLEU-2 score.
        - bleu3 (float): BLEU-3 score.
        - bleu4 (float): BLEU-4 score.
        - avg_jaccard (float): Average Jaccard similarity score.
        - gen_length (float): Average length of generated dialogues.
    """
    if len(references) != len(predictions):
        raise ValueError("The number of references and predictions must be the same.")

    # Compute ROUGE scores
    rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    # Compute BLEU score
    bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    
    # Compute average Jaccard similarity
    jaccard_scores = [calculate_jaccard_similarity_for_texts(ref, pred) for ref, pred in zip(references, predictions)]
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    # Compute average generation length
    gen_length = sum(len(pred.split()) for pred in predictions) / len(predictions) if predictions else 0.0

    # Combine all metrics into a single dictionary
    metrics = {
        metrics_prefix + 'rouge1': rouge_results['rouge1'],
        metrics_prefix + 'rouge2': rouge_results['rouge2'],
        metrics_prefix + 'rougeL': rouge_results['rougeL'],
        metrics_prefix + 'rougeLsum': rouge_results['rougeLsum'],
        metrics_prefix + 'bleu': bleu_results['bleu'],
        metrics_prefix + 'bleu1': bleu_results['precisions'][0],
        metrics_prefix + 'bleu2': bleu_results['precisions'][1],
        metrics_prefix + 'bleu3': bleu_results['precisions'][2],
        metrics_prefix + 'bleu4': bleu_results['precisions'][3],
        metrics_prefix + 'avg_jaccard': avg_jaccard,
        metrics_prefix + 'gen_length': gen_length
    }

    return metrics

def save_preds_to_file(refs: list, preds: list, filename: str) -> None:
    """
    Saves references and predictions to a specified file.

    Args:
        refs (list): List of reference dialogues.
        preds (list): List of predicted dialogues.
        filename (str): The filename to save the references and predictions.
    """
    data = {"references": refs, "predictions": preds}
    if not PREDICTIONS_DIR.exists():
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(PREDICTIONS_DIR / filename, index=False)

def get_compute_metrics_function_for_stories(
        tokenizer: AutoTokenizer,
        metrics_prefix: str = "",
        save_preds: bool = False,
        save_preds_filename: str | None = None
    ) -> callable:
    """
    Create a function to compute metrics for story-to-dialogue conversion tasks that can be used with the Hugging Face Trainer.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used for decoding model outputs.
        metrics_prefix (str | None): Optional prefix to add to metric names.
        save_preds (bool): Whether to save predictions to a file.
        save_preds_filename (str | None): Filename to save predictions if `save_preds` is True.

    Returns:
        callable: A function that computes metrics given an EvalPrediction object.
    """
    if save_preds and (save_preds_filename is None or save_preds_filename.strip() == ""):
        raise ValueError("save_preds_filename must be a non-empty string when save_preds is True.")
    if save_preds and not save_preds_filename.endswith('.csv'):
        raise ValueError("save_preds_filename must have a .csv extension.")

    def compute_metrics(p: EvalPrediction) -> dict:
        """
        Computes metrics for the given EvalPrediction object. This function is intended to be used with the Hugging Face Trainer.

        Args:
            p (EvalPrediction): An object containing model predictions and label IDs.
        """
        preds = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
        refs = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
        if save_preds:
            save_preds_to_file(refs, preds, save_preds_filename)
        return compute_metrics_for_stories(refs, preds, metrics_prefix)

    return compute_metrics