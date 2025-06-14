import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW

device = torch.device("cuda")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.texts)

#### model load ####
model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

#### data load ####
train_texts = ["This is the first sentence.", "This is the second sentence."]
train_labels = [0, 1]
train_dataset = MyDataset(train_texts, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#### train ####
epochs = 10
lr = 2e-5
optimizer = AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for texts, labels in train_dataloader:
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            input_ids.append(encoded["input_ids"].squeeze())
            attention_masks.append(encoded["attention_mask"].squeeze())

        input_ids = torch.stack(input_ids).to(device)
        attention_masks = torch.stack(attention_masks).to(device)
        labels = torch.tensor(labels).to(device)

        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")

#### save model ####
model.save_pretrained("path/to/save/model")
tokenizer.save_pretrained("path/to/save/tokenizer")