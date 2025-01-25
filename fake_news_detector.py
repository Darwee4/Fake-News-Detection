import kagglehub
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import pandas as pd
import numpy as np

# Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
print(f"Dataset downloaded to: {path}")

# Load dataset
df = pd.read_csv(f"{path}/WELFake_Dataset.csv")
df = df.drop(columns=['Unnamed: 0'])  # Remove index column
df = df.dropna(subset=['text'])  # Drop rows with missing text
df['text'] = df['text'].astype(str)  # Ensure all text is string type
# The dataset already contains labels: 1 for fake, 0 for real

# Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data(texts, labels, max_length=128):
    # Ensure texts is a list of strings
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    if not isinstance(texts, list):
        texts = [texts]
    
    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return FakeNewsDataset(encodings, labels)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

train_dataset = preprocess_data(train_texts.tolist(), train_labels.tolist())
val_dataset = preprocess_data(val_texts.tolist(), val_labels.tolist())

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train model
print("Training model...")
trainer.train()

# Evaluate model
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save model
model.save_pretrained('./fake_news_model')
tokenizer.save_pretrained('./fake_news_model')
print("Model saved to ./fake_news_model")

# Prediction function
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "Fake" if torch.argmax(probs).item() == 1 else "Real"

# Test prediction
test_text = "This is a sample news article to test the model."
print(f"Prediction for test text: {predict_news(test_text)}")
