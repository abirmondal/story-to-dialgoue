"""
story_eval.py

This module contains functions to calculate various metrics for evaluating story-to-dialogue conversion tasks.
"""

import evaluate
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, EvalPrediction
from config.dir import PREDICTIONS_DIR

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
bertscore = evaluate.load('bertscore')

def calculate_distinct_n_grams(text: str, n: int) -> float:
    """
    Calculates the Distinct-n metric for a given text.

    Args:
        text (str): The input text.
        n (int): The n-gram size.

    Returns:
        float: The Distinct-n score.
    """
    if not text:
        return 0.0
    
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    n_grams = set()
    for i in range(len(tokens) - n + 1):
        n_gram = tuple(tokens[i:i + n])
        n_grams.add(n_gram)
    distinct_n = len(n_grams) / (len(tokens) - n + 1)
    return distinct_n

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

def compute_metrics_for_stories(
        references: list,
        predictions: list,
        metrics_prefix: str = "",
        include_bertscore: bool = False,
        bertscore_lang: str = 'en',
        bertscore_model_type: str = 'distilbert-base-uncased',
        bertscore_verbose: bool = False
    ) -> dict:
    """
    Computes various evaluation metrics for story-to-dialogue conversion.

    Args:
        references (list): List of reference dialogues.
        predictions (list): List of predicted dialogues.
        metrics_prefix (str): Optional prefix to add to metric names.
        include_bertscore (bool): Whether to include BERTScore in the metrics. Default is False.
        bertscore_lang (str): Language for BERTScore computation. Default is 'en'. Required if `include_bertscore` is True.
        bertscore_model_type (str): Model type for BERTScore computation. Default is 'distilbert-base-uncased'. Required if `include_bertscore` is True.
        bertscore_verbose (bool): Whether to enable verbose output for BERTScore computation. Default is False. Required if `include_bertscore` is True.
        
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
        - meteor (float): METEOR score.
        - distinct_1 (float): Distinct-1 score.
        - distinct_2 (float): Distinct-2 score.
        - distinct_3 (float): Distinct-3 score.
        - avg_distinct_1 (float): Average Distinct-1 score.
        - avg_distinct_2 (float): Average Distinct-2 score.
        - avg_distinct_3 (float): Average Distinct-3 score.
        - avg_jaccard (float): Average Jaccard similarity score.
        - gen_length (float): Average length of generated dialogues.
        - bertscore_precision (float, optional): BERTScore precision (if include_bertscore is True).
        - bertscore_recall (float, optional): BERTScore recall (if include_bertscore is True).
        - bertscore_f1 (float, optional): BERTScore F1 (if include_bertscore is True).
    """
    if len(references) != len(predictions):
        raise ValueError("The number of references and predictions must be the same.")
    
    if include_bertscore and not bertscore_lang:
        raise ValueError("bertscore_lang must be provided when include_bertscore is True.")
    if include_bertscore and not bertscore_model_type:
        raise ValueError("bertscore_model_type must be provided when include_bertscore is True.")
    if include_bertscore and bertscore_verbose is None:
        raise ValueError("bertscore_verbose must be provided when include_bertscore is True.")

    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True)
    
    # Compute BLEU score
    bleu_results = bleu.compute(predictions=predictions, references=[
                                [ref] for ref in references])

    # Compute METEOR score
    meteor_results = meteor.compute(
        predictions=predictions, references=references)
    
    # Compute Distinct-n scores
    distinct_1_scores = [calculate_distinct_n_grams(pred, 1) for pred in predictions]
    distinct_2_scores = [calculate_distinct_n_grams(pred, 2) for pred in predictions]
    distinct_3_scores = [calculate_distinct_n_grams(pred, 3) for pred in predictions]

    # Compute average Distinct-n scores
    avg_distinct_1 = sum(distinct_1_scores) / \
        len(distinct_1_scores) if distinct_1_scores else 0.0
    avg_distinct_2 = sum(distinct_2_scores) / \
        len(distinct_2_scores) if distinct_2_scores else 0.0
    avg_distinct_3 = sum(distinct_3_scores) / \
        len(distinct_3_scores) if distinct_3_scores else 0.0

    # Compute average Jaccard similarity
    jaccard_scores = [calculate_jaccard_similarity_for_texts(
        ref, pred) for ref, pred in zip(references, predictions)]
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    # Compute average generation length
    gen_length = sum(len(pred.split()) for pred in predictions) / \
        len(predictions) if predictions else 0.0
    
    # Compute BERTScore if required
    if include_bertscore:
        bertscore_results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang=bertscore_lang,
            model_type=bertscore_model_type,
            verbose=bertscore_verbose
        )
        metrics_prefix_bertscore = {
            metrics_prefix + 'bertscore_precision': np.mean(bertscore_results['precision']),
            metrics_prefix + 'bertscore_recall': np.mean(bertscore_results['recall']),
            metrics_prefix + 'bertscore_f1': np.mean(bertscore_results['f1'])
        }

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
        metrics_prefix + 'meteor': meteor_results['meteor'],
        metrics_prefix + 'distinct_1': distinct_1_scores,
        metrics_prefix + 'distinct_2': distinct_2_scores,
        metrics_prefix + 'distinct_3': distinct_3_scores,
        metrics_prefix + 'avg_distinct_1': avg_distinct_1,
        metrics_prefix + 'avg_distinct_2': avg_distinct_2,
        metrics_prefix + 'avg_distinct_3': avg_distinct_3,
        metrics_prefix + 'avg_jaccard': avg_jaccard,
        metrics_prefix + 'gen_length': gen_length
    }

    if include_bertscore:
        metrics.update(metrics_prefix_bertscore)

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
        save_preds_filename: str | None = None,
        include_bertscore: bool = False,
        bertscore_lang: str = 'en',
        bertscore_model_type: str = 'distilbert-base-uncased',
        bertscore_verbose: bool = False
    ) -> callable:
    """
    Create a function to compute metrics for story-to-dialogue conversion tasks that can be used with the Hugging Face Trainer.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used for decoding model outputs.
        metrics_prefix (str | None): Optional prefix to add to metric names.
        save_preds (bool): Whether to save predictions to a file.
        save_preds_filename (str | None): Filename to save predictions if `save_preds` is True.
        bertscore (bool): Whether to include BERTScore in the metrics. Default is False.

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
        preds = np.where(p.predictions != -100, p.predictions,
                         tokenizer.pad_token_id)
        labels = np.where(p.label_ids != -100, p.label_ids,
                          tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        if save_preds:
            save_preds_to_file(decoded_labels, decoded_preds, save_preds_filename)
        return compute_metrics_for_stories(
            decoded_labels, decoded_preds, metrics_prefix, 
            include_bertscore, bertscore_lang, bertscore_model_type, bertscore_verbose
        )

    return compute_metrics