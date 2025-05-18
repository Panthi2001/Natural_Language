### 📰 Fake News Detection & Abstractive Summarization with Transformers
Natural_Language
Senior Year Project
Course: CSCE 4290 - Natural Language Processing
Professor: Zeenat Tariq
Date: Spring 2025
Team Members (Group 15): Aavash Neupane, Kushal Panthi, Siddhartha Pudasaini, Suraj Varne Sheela
GitHub Repo: Natural_Language

## Project Title:
Fake News Detection and Summarization

## Project Overview
This project implements a complete Natural Language Processing pipeline to identify fake news articles and generate concise, abstractive summaries of them using advanced ML and deep learning techniques.

We approached this task in two stages:

## Fake News Classification

Applied multiple supervised learning models (SVM, Naive Bayes, Logistic Regression) with TF-IDF vectorization

Achieved up to 86.4% classification accuracy with F1-score of 0.86

Evaluated performance using confusion matrices and classification reports

## Abstractive Summarization

Utilized BERTSum via Hugging Face Transformers to summarize fake news articles

Fine-tuned a BART model for generating summaries with over 82% semantic retention

Evaluated summaries using ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum

## Tools Used
Anaconda: Open-source Python distribution

Jupyter Notebook: Web-based interactive coding environment

Google Colab: Cloud-based development environment

Git & GitHub: Version control and collaboration

GroupMe, Email: Communication tools

 ## Libraries & Frameworks
NumPy: Numerical operations

Pandas: Data manipulation

Matplotlib, Seaborn: Visualization

NLTK, re: Natural language preprocessing

Scikit-learn: Machine learning models

TF-IDF: Text vectorization

## Dataset
https://www.kaggle.com/datasets/mdepak/fakenewsnet
A pre-provided dataset containing labeled real and fake news articles.


kaggle.json has api keys and data sets is imported using python script

## 📊 Results (Highlights)
Task	Metric	Value
Fake News Classification	Accuracy	86.4%
Classification	F1-Score	0.86
Summarization	ROUGE-Lsum	+18.2% (post-tuning)
Summarization	Article Reduction	~63% avg

