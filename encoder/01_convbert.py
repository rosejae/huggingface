from transformers import ConvBertTokenizer, ConvBertForMaskedLM
import torch

def predict_next_word(sentence):
    model_name = "YituTech/conv-bert-base"
    tokenizer = ConvBertTokenizer.from_pretrained(model_name)
    model = ConvBertForMaskedLM.from_pretrained(model_name)

    tokens = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt")
    masked_index = torch.where(tokens == tokenizer.mask_token_id)[1].tolist()[0]

    with torch.no_grad():
        outputs = model(tokens)
        predictions = outputs.logits[0, masked_index]

    top_predictions = torch.topk(predictions, k=5)
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_predictions.indices.tolist())
    probabilities = top_predictions.values.exp().tolist()

    return predicted_tokens, probabilities

#### data load ####
sentence = "I want to [MASK] a car"

#### inference ####
predicted_tokens, probabilities = predict_next_word(sentence)
for token, prob in zip(predicted_tokens, probabilities):
    print(f"Token: {token}, Probability: {prob}")