import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification

#### model load ####
model_name = "xlnet-base-cased"
tokenizer = XLNetTokenizer.from_pretrained(l)
model = XLNetForSequenceClassification.from_pretrained(l, num_labels=2)

#### model inference ####
text = "The dog is really cute."

inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)

outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

labels = ['Negative', 'Positive']  
predicted_label = labels[predicted_class]

print(f"Text: {text}")
print(f"Predicted Label: {predicted_label}")
