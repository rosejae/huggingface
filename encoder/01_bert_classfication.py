########################
#### classification ####
########################

import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device('cuda')

def classify_text(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,           
        max_length=128,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.to('cuda')(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze(dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities

#### model load ####
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

#### inference ###
text_to_classify = "This is an example sentence."
predicted_class, probabilities = classify_text(text_to_classify)

print(f"Predicted class: {predicted_class}")
print("Probabilities:")
for i, prob in enumerate(probabilities):
    print(f"Class {i}: {prob.item()}")