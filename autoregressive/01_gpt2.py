import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#### model load ####
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#### inference (greedy) ####
input_ids = tokenizer.encode('My name is Hasnain and I am a software engineer.', return_tensors='tf')
greedy_output = model.generate(input_ids, max_length=50)
print(f"Output: {tokenizer.decode(greedy_output[0])}")

#### inference (beam) ####
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)
print(f"Output: {tokenizer.decode(beam_output[0], skip_special_tokens=True)}")




