import torch
from transformers import XLMProphetNetTokenizer, XLMProphetNetForCausalLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

device = torch.device("cuda")

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

### model load ###
model_name = 'microsoft/xprophetnet-large-wiki100-cased'
model = XLMProphetNetForCausalLM.from_pretrained(model_name).to(device)
tokenizer = XLMProphetNetTokenizer.from_pretrained(model_name)

### data load ###
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:100]') 
texts = dataset['text']
custom_dataset = CustomDataset(texts, tokenizer, max_length)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

### train ###
batch_size = 5 # really huge model, if it was batch_size=2 or 3, it would not run
max_length = 128
num_epochs = 3
learning_rate = 2e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

### inference ###
model.eval()
example_prompt = "Once upon a time"
input = tokenizer(example_prompt, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model(**input)
    logits = outputs.logits
    predicted_label = torch.argmax(logits[0], dim=1)
    generated_text = tokenizer.decode(predicted_label, skip_special_tokens=True)
    print(f'Example Prompt: {example_prompt}')
    print(f'Generated Text: {generated_text}')

