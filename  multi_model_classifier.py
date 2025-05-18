# multi_model_classifier.py
# Comprehensive script to train, evaluate, and save multiple classification models
# for fake news detection using TF-IDF features.

import os
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils_preprocess import load_config, load_dataset, preprocess_dataframe, build_tfidf_vectorizer, vectorize_texts

def train_and_evaluate_classifiers(config):
    """
    Train SVM, Naive Bayes, and Logistic Regression, then
    evaluate with detailed reports and confusion matrices.
    """
    # Load and preprocess data
    df = load_dataset(config['dataset']['fake_news_csv'])
    df = preprocess_dataframe(df)
    X = df['cleaned']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['classification']['test_size'], random_state=config['classification']['random_seed']
    )

    # Build TF-IDF
    vectorizer = build_tfidf_vectorizer(config)
    X_train_tfidf, X_test_tfidf = vectorize_texts(vectorizer, X_train, X_test)

    # Save vectorizer
    os.makedirs(config['output']['classifier_model_dir'], exist_ok=True)
    joblib.dump(vectorizer, os.path.join(config['output']['classifier_model_dir'], "tfidf_vectorizer.pkl"))

    # Train each classifier
    classifiers = {
        "SVM": SVC(kernel=config['classification']['svm']['kernel'],
                   C=config['classification']['svm']['C'],
                   probability=True,
                   random_state=config['classification']['random_seed']),
        "Naive Bayes": MultinomialNB(alpha=config['classification']['naive_bayes']['alpha']),
        "Logistic Regression": LogisticRegression(
            C=config['classification']['logistic_regression']['C'],
            max_iter=config['classification']['logistic_regression']['max_iter'],
            random_state=config['classification']['random_seed']
        )
    }

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_tfidf, y_train)
        joblib.dump(clf, os.path.join(config['output']['classifier_model_dir'], f"{name.replace(' ', '_').lower()}_model.pkl"))
        print(f"{name} saved to disk.")

        # Evaluate
        y_pred = clf.predict(X_test_tfidf)
        _print_report(name, y_test, y_pred)

        # Plot confusion matrix
        _plot_confusion_matrix(name, y_test, y_pred)

def _print_report(model_name, y_true, y_pred):
    """
    Print classification report and metrics to console.
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average=config['evaluation']['f1_average'])
    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score ({config['evaluation']['f1_average']}): {f1:.4f}")
    print("Classification Report:")
    print(metrics.classification_report(y_true, y_pred, digits=4))

def _plot_confusion_matrix(model_name, y_true, y_pred):
    """
    Plot and save confusion matrix as PNG file.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(config['output']['classifier_model_dir'], f"{model_name.replace(' ', '_').lower()}_confusion.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to {output_path}.")

if __name__ == "__main__":
    config = load_config()
    train_and_evaluate_classifiers(config)
