# model_eval.py
# Evaluation utilities for measuring ROUGE metrics for summarization
# and computing F1/Accuracy for classification.

import os
import yaml
import torch
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from sklearn import metrics
from transformers import BertTokenizer, BartForConditionalGeneration

from utils_preprocess import load_config

def compute_rouge_scores(references, candidates, metrics_list=None):
    """
    Compute ROUGE scores (rouge1, rouge2, rougeL, rougeLsum) 
    between lists of reference summaries and candidate summaries.
    Returns a dictionary of averaged scores.
    """
    if metrics_list is None:
        metrics_list = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(metrics_list, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    # Convert to simple dict: {metric: {precision, recall, fmeasure}}
    avg_scores = {
        metric: {
            'precision': result[metric].mid.precision,
            'recall': result[metric].mid.recall,
            'fmeasure': result[metric].mid.fmeasure
        }
        for metric in metrics_list
    }
    return avg_scores

def evaluate_summarizer_on_dataset(config, input_csv, output_dir="eval_outputs"):
    """
    Load articles and reference summaries from a CSV, run summarizer, and compute ROUGE.
    Assumes CSV has columns: 'article' and 'reference_summary'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load pretrained model
    tokenizer = BertTokenizer.from_pretrained(config['summarization']['model_name'])
    model = BartForConditionalGeneration.from_pretrained(config['summarization']['model_name']).to(config['summarization']['device'])

    df = pd.read_csv(input_csv)
    references = df['reference_summary'].tolist()
    articles = df['article'].tolist()
    candidates = []

    for idx, art in enumerate(articles):
        inputs = tokenizer(art, return_tensors="pt", max_length=config['summarization']['max_input_length'], truncation=True).to(config['summarization']['device'])
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=config['summarization']['num_beams'],
            max_length=config['summarization']['max_summary_length'],
            repetition_penalty=config['summarization']['repetition_penalty'],
            length_penalty=config['summarization']['length_penalty'],
            early_stopping=config['summarization']['early_stopping']
        )
        cand = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        candidates.append(cand)

    rouge_scores = compute_rouge_scores(references, candidates, config['evaluation']['rouge_metrics'])

    # Save scores to CSV
    score_df = pd.DataFrame.from_dict({
        metric: [rouge_scores[metric]['precision'], rouge_scores[metric]['recall'], rouge_scores[metric]['fmeasure']]
        for metric in rouge_scores
    }, orient='index', columns=['precision', 'recall', 'fmeasure'])
    score_df.to_csv(os.path.join(output_dir, "rouge_scores.csv"))

    print("ROUGE evaluation complete. Scores:")
    print(score_df)
    return rouge_scores

def classification_metrics(y_true, y_pred):
    """
    Compute and print accuracy, F1, precision, recall, and confusion matrix for classification.
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=config['evaluation']['f1_average'])
    precision = metrics.precision_score(y_true, y_pred, average=config['evaluation']['f1_average'])
    recall = metrics.recall_score(y_true, y_pred, average=config['evaluation']['f1_average'])
    cm = metrics.confusion_matrix(y_true, y_pred)

    print("Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision ({config['evaluation']['f1_average']}): {precision:.4f}")
    print(f"Recall ({config['evaluation']['f1_average']}): {recall:.4f}")
    print(f"F1-score ({config['evaluation']['f1_average']}): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    config = load_config()
    # Example usage:
    # Assume we have a CSV "data/summ_eval.csv" with columns 'article' and 'reference_summary'
    rouge_results = evaluate_summarizer_on_dataset(config, "data/summ_eval.csv")
    print("Aggregated ROUGE scores:", rouge_results)
