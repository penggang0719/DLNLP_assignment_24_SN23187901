import os
import re
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_imdb(data_dir, split):
    data, labels = [], []
    for label in ["pos", "neg"]:
        folder_name = os.path.join(data_dir, split, label)
        for file_name in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data.append(f.read().strip())
                labels.append(0 if label == "neg" else 1)
    return {"text": data, "label": labels}


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    # Remove non-word and non-whitespace (punctuation & special characters)
    text = re.sub(r'\W+|\s+', ' ', text)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def clean_all_texts(data):
    return [clean_text(text) for text in data["text"]]


# Load IMDB data
def load_imdb_data():
    with open("A/Preprocessed_Data/clean_train_data.json", "r") as f:
        clean_train_data = json.load(f)
    with open("A/Preprocessed_Data/clean_test_data.json", "r") as f:
        clean_test_data = json.load(f)

    train_token = torch.load("A/Preprocessed_Data/train_token.pt")
    test_token = torch.load("A/Preprocessed_Data/test_token.pt")

    return clean_train_data, clean_test_data, train_token, test_token


def select_test(data, num_samples=1250, seed=42):
    # Separate data by label
    neg_indices = [i for i, label in enumerate(data["label"]) if label == 0]
    pos_indices = [i for i, label in enumerate(data["label"]) if label == 1]

    # Randomly select samples from each class
    random.seed(seed)
    selected_neg_indices = random.sample(neg_indices, num_samples)
    selected_pos_indices = random.sample(pos_indices, num_samples)

    # Combine selected indices
    selected_indices = selected_neg_indices + selected_pos_indices
    random.shuffle(selected_indices)

    # Create new data
    selected_data = {"text": [data["text"][i] for i in selected_indices],
                     "label": [data["label"][i] for i in selected_indices]}
    
    return selected_data


class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train(model, data_loader, optimizer, scheduler, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0
    correct_predictions = 0
    optimizer.zero_grad()
    for step, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == batch['labels'])
            scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    device = next(model.parameters()).device  # 获取模型所在设备
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}  # 确保数据在正确的设备上
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += (preds == batch['labels']).sum().item()
            predictions.append(preds)
            true_labels.append(batch['labels'])
    # 一次性将所有结果转移到CPU并转换为numpy数组
    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()
    accuracy = correct_predictions / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy, predictions, true_labels


def plot_length_distribution(train_lengths, test_lengths,filepath):
    plt.figure(figsize=(10, 5))
    plt.hist(train_lengths, bins=30, alpha=0.5, label='Train')
    plt.hist(test_lengths, bins=30, alpha=0.5, label='Test')
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(filepath)
    plt.show()


def plot_confusion_matrix(true_labels, predictions, filepath):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.show()