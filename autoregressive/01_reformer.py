from transformers import AutoTokenizer, ReformerForQuestionAnswering
from transformers import ReformerModelWithLMHead, ReformerTokenizer
import torch

#### model load (ReformerForQuestionAnswering) ####
model_name = "google/reformer-crime-and-punishment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ReformerForQuestionAnswering.from_pretrained(model_name)

#### inference ####
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

#### ReformerModelWithLMHead (Deprecated) ####
# model_name = "google/reformer-crime-and-punishment"
# model = ReformerModelWithLMHead.from_pretrained(model_name)
# tokenizer = ReformerTokenizer.from_pretrained(model_name)

# input_ids = tokenizer.encode("A few months later", return_tensors="pt")
# answer_ids = model.generate(input_ids, do_sample=True,temperature=0.7, max_length=100)

# print(f"generated answer: {tokenizer.decode(answer_ids[0]}")


