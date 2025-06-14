import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

device = torch.device("cuda")

#### model load ###
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

#### data load ####
input_text = "The capital of Korea is <mask>."
input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt').to(device)
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

#### inference ####
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

predicted_token_ids = torch.argmax(predictions[0, masked_index], dim=1).tolist()
predicted_tokens = tokenizer.batch_decode(predicted_token_ids)
print(f"predicted word: {Predicted_token}")