################################
#########Text-to-Speech#########
################################

import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
import soundfile as sf
from IPython.display import Audio

### model load ###
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

### vocoder load ###
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

### input load (text) ###
inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")

### inference ###
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
sf.write("tts_example.wav", speech.numpy(), samplerate=16000)
# Audio(speech, rate=16000)



################################
########Speech-to-Speech########
################################

import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
from transformers import SpeechT5HifiGan
import soundfile as sf
from IPython.display import Audio

### model load ###
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

### vocoder load ###
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[1236]["xvector"]).unsqueeze(0)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

### input load (speech) ###
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
example = dataset.sort("id")[40] 
sampling_rate = dataset.features["audio"].sampling_rate
# Audio(example["audio"]["array"], rate=16000)

### inference ###
inputs = processor(audio=example["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
sf.write("speech_converted.wav", speech.numpy(), samplerate=16000)
# Audio(speech, rate=16000)



############################################
########Automatic speech recognition########
##################pipeline##################
############################################

from transformers import pipeline
generator = pipeline(task="automatic-speech-recognition", model="microsoft/speecht5_asr")

transcription = generator(example["audio"]["array"])
print(f"recognized speech: {transcription["text"]}")

############################################
########Automatic speech recognition########
###############using the model##############
############################################

from transformers import SpeechT5ForSpeechToText

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

sampling_rate = dataset.features["audio"].sampling_rate
inputs = processor(audio=example["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

predicted_ids = model.generate(**inputs, max_length=100)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(f"recognized speech: {transcription[0]}")