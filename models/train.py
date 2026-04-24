import pandas as pd
import numpy as np
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("data/data.csv")

# Keep only required columns
df = df[["text", "sentiment"]].dropna()

# -------------------------
# 2. Clean labels
# -------------------------
df["sentiment"] = df["sentiment"].str.lower().str.strip()

label_map = {
    "negative": 0,
    "positive": 1
}

df["label"] = df["sentiment"].map(label_map)

df = df[["text", "label"]].dropna()

# -------------------------
# 3. Convert dataset
# -------------------------
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# -------------------------
# 4. Model + tokenizer
# -------------------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# -------------------------
# 5. Training arguments (SAFE VERSION)
# -------------------------
training_args = TrainingArguments(
    output_dir="model/",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)

# -------------------------
# 6. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# -------------------------
# 7. Train model
# -------------------------
print("🚀 Training started...")
trainer.train()
print("✅ Training completed")

# -------------------------
# 8. Save model
# -------------------------
trainer.save_model("model/saved_model")
tokenizer.save_pretrained("model/saved_model")

