#####################
#### fine tuning ####
#####################

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

device = torch.device("cuda")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = self.labels[idx]
        if isinstance(label, torch.Tensor):
            item["labels"] = label
        else:
            item["labels"] = torch.tensor(label)
        return item

    def __len__(self):
        return len(self.labels)

### model load ###
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

#### data load ###
data_name = "imdb"
dataset = load_dataset(data_name)

#### data preparation ####
subset = dataset["train"].select(range(2000))
split_dataset = subset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]

train_encodings = tokenizer(
    train_data["text"],
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_data["text"],
    truncation=True,
    padding=True,
    max_length=128
)

train_labels = torch.tensor(train_data["label"])
val_labels = torch.tensor(val_data["label"])
train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

#### train ####
num_epochs = 5

optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * num_epochs
)

model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / len(val_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

#### save model ####
model.save_pretrained("./xlm_roberta/model")
tokenizer.save_pretrained("./xlm_roberta/tokenizer")

#####################
#####################
#####################

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

#### pretrained model load ####
model = XLMRobertaForSequenceClassification.from_pretrained("./xlm_roberta/model")
tokenizer = XLMRobertaTokenizer.from_pretrained("./xlm_roberta/tokenizer")

#### data load ####
data_name = "imdb"
dataset = load_dataset(data_name)
subset = dataset["train"].select(range(2000))

split_dataset = subset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]

sentence_example = val_data["text"][0]
inputs = tokenizer(sentence_example, truncation=True, padding=True, return_tensors="pt")

#### inference ####
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    
print(f"predicted label: {predicted_label}")
print(f"true label: {val_data['label'][0]}")