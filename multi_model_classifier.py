# multi_model_classifier.py
# Train and evaluate multiple classifiers (SVM, Naive Bayes, Logistic Regression)
# using TF-IDF features for fake news classification.

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from utils_preprocess import load_config, load_dataset, preprocess_dataframe, build_tfidf_vectorizer, vectorize_texts

def train_and_evaluate_classifiers(config):
    """
    Train SVM, Naive Bayes, and Logistic Regression on the FakeNewsNet dataset.
    Reports classification metrics and saves models to disk.
    """
    # Load and preprocess dataset
    df = load_dataset(config['dataset']['fake_news_csv'])
    df = preprocess_dataframe(df)

    X = df['cleaned']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['classification']['test_size'], random_state=config['classification']['random_seed']
    )

    # Build TF-IDF vectorizer
    vectorizer = build_tfidf_vectorizer(config)
    X_train_tfidf, X_test_tfidf = vectorize_texts(vectorizer, X_train, X_test)

    # Ensure output directory exists
    os.makedirs(config['output']['classifier_model_dir'], exist_ok=True)
    joblib.dump(vectorizer, os.path.join(config['output']['classifier_model_dir'], "tfidf_vectorizer.pkl"))

    # 1) SVM Classifier
    svm_cfg = config['classification']['svm']
    svm_clf = SVC(
        kernel=svm_cfg['kernel'],
        C=svm_cfg['C'],
        probability=True,
        random_state=config['classification']['random_seed']
    )
    svm_clf.fit(X_train_tfidf, y_train)
    joblib.dump(svm_clf, os.path.join(config['output']['classifier_model_dir'], "svm_model.pkl"))
    _evaluate_and_report("SVM", svm_clf, X_test_tfidf, y_test)

    # 2) Naive Bayes Classifier
    nb_cfg = config['classification']['naive_bayes']
    nb_clf = MultinomialNB(alpha=nb_cfg['alpha'])
    nb_clf.fit(X_train_tfidf, y_train)
    joblib.dump(nb_clf, os.path.join(config['output']['classifier_model_dir'], "nb_model.pkl"))
    _evaluate_and_report("Naive Bayes", nb_clf, X_test_tfidf, y_test)

    # 3) Logistic Regression Classifier
    lr_cfg = config['classification']['logistic_regression']
    lr_clf = LogisticRegression(
        C=lr_cfg['C'],
        max_iter=lr_cfg['max_iter'],
        random_state=config['classification']['random_seed']
    )
    lr_clf.fit(X_train_tfidf, y_train)
    joblib.dump(lr_clf, os.path.join(config['output']['classifier_model_dir'], "lr_model.pkl"))
    _evaluate_and_report("Logistic Regression", lr_clf, X_test_tfidf, y_test)

def _evaluate_and_report(model_name, clf, X_test, y_test):
    """
    Helper function to compute metrics and print a detailed classification report.
    """
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average=config['evaluation']['f1_average'])
    cm = metrics.confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score ({config['evaluation']['f1_average']}): {f1:.4f}")
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    config = load_config()
    train_and_evaluate_classifiers(config)
