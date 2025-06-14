#####################
#### fine-tuning ####
#####################

from datasets import load_dataset
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

def compute_labels(examples):
    labels = examples['label']
    if isinstance(labels, list):  
        return {"labels": [0 if label == :"neg" else 1 for label in labels]}
    else:
        return {"labels": [0 if labels == "neg" else 1]}

### model load ###
model_name = "flaubert/flaubert_base_cased"
tokenizer = FlaubertTokenizer.from_pretrained(model_name)
model = FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=2)

### data load ###
dataset = load_dataset("imdb", split="train[:1000]")
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.map(compute_labels)

### train ###
training_args = TrainingArguments(
    "test-trainer",
    # evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
)

trainer.train()

#####################
##### inference #####
#####################

device = torch.device("cuda")

#### data load ####
sentences = ["I love this movie!"]
inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

#### evaluation ####
outputs = model(**inputs)
logits = outputs.logits
probs = torch.nn.functional.softmax(logits, dim=-1)
predictions = torch.argmax(probs, dim=-1)
print(f"predictions: {predictions}")