from transformers import AutoProcessor, AutoModel
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor(
    text=["Hello, my name is Harish. They call me Doniker, uh — and I like Red Dead Redemption 2. [laughs] But I also like other games such as GTA 5 AAAAAAAAAAAAA Kaiose pro max"],
    return_tensors="pt", voice_preset=voice_preset
)

speech_values = model.generate(**inputs, do_sample=True).to(device)

import scipy

sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())