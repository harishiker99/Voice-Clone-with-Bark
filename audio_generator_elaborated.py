from transformers import AutoProcessor, AutoModel
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark").to(device)

voice_preset = "v2/en_speaker_6"

inputs = processor(
    text=["Hello, my name is Harish Nair and I need assistance with my course selection and co-op work term!"],
    return_tensors="pt", voice_preset=voice_preset
).to(device)

speech_values = model.generate(**inputs, do_sample=True).to(device)

import scipy  # noqa: E402

sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("mdti.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())