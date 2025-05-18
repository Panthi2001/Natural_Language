# bert_summarizer.py
# Advanced abstractive summarization script using BART (BERT-based) from Hugging Face.
# Includes training loop stubs and generation utilities.

import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from utils_preprocess import load_config, clean_text

class FakeNewsSummarizationDataset(Dataset):
    """
    Custom PyTorch Dataset for fake news summarization.
    Expects a CSV with columns 'article' and 'summary'.
    """
    def __init__(self, articles, summaries, tokenizer, max_input_len, max_output_len):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = clean_text(self.articles[idx])
        summary = clean_text(self.summaries[idx])

        # Tokenize inputs and outputs
        inputs = self.tokenizer(
            article,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            summary,
            max_length=self.max_output_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = targets.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_summarizer(config, train_csv, val_csv, output_dir):
    """
    Fine-tune BART model for abstractive summarization on FakeNewsNet.
    """
    # Load configuration
    model_name = config['summarization']['model_name']
    max_input_len = config['summarization']['max_input_length']
    max_output_len = config['summarization']['max_summary_length']
    device = torch.device(config['summarization']['device'] if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Load data
    df_train = pd.read_csv(train_csv)  # Columns: 'article','summary'
    df_val = pd.read_csv(val_csv)

    train_articles, train_summaries = df_train['article'].tolist(), df_train['summary'].tolist()
    val_articles, val_summaries = df_val['article'].tolist(), df_val['summary'].tolist()

    # Create datasets and loaders
    train_dataset = FakeNewsSummarizationDataset(train_articles, train_summaries, tokenizer, max_input_len, max_output_len)
    val_dataset = FakeNewsSummarizationDataset(val_articles, val_summaries, tokenizer, max_input_len, max_output_len)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop stub (detailed training commented for brevity)
    model.train()
    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        epoch_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Average training loss for epoch {epoch+1}: {avg_epoch_loss:.4f}")

        # (Optional) Add validation loop here with no_grad to compute validation loss/metrics

    # Save model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def generate_summary(config, model_dir, article_text):
    """
    Load a fine-tuned model from disk and generate a summary for a single article.
    """
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir).to(config['summarization']['device'])
    model.eval()

    article_clean = clean_text(article_text)

    inputs = tokenizer(
        article_clean,
        return_tensors="pt",
        max_length=config['summarization']['max_input_length'],
        truncation=True
    ).to(config['summarization']['device'])

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=config['summarization']['num_beams'],
        max_length=config['summarization']['max_summary_length'],
        repetition_penalty=config['summarization']['repetition_penalty'],
        length_penalty=config['summarization']['length_penalty'],
        early_stopping=config['summarization']['early_stopping']
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    config = load_config()

    # Example: Fine-tune on training and validation CSVs
    train_summarizer(
        config,
        train_csv="data/train_summaries.csv",
        val_csv="data/val_summaries.csv",
        output_dir=config['output']['summarizer_model_dir']
    )

    # Example: Generate a summary
    example_text = """
    In recent years, fake news has proliferated on social media, creating confusion among the public. 
    Researchers are now leveraging state-of-the-art neural architectures to automatically detect and 
    summarize misleading articles. This study uses BERT-based encoders and decoders to produce concise, 
    accurate summaries of fake news content for faster fact-checking.
    """
    summary = generate_summary(config, config['output']['summarizer_model_dir'], example_text)
    print("Generated Summary:\n", summary)
