# Author: @echu2
# Extract NLL from MusicGen

import torch
import librosa
from functools import cache
from transformers import AutoProcessor, MusicgenForConditionalGeneration

class NLLExtractor:
    def __init__(self, model_name="facebook/musicgen-small"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_name
        ).to(self.device)

        # turn off text conditioning
        self.model.config.use_text_conditioning = False        
        self.sr = self.model.config.audio_encoder.sampling_rate

        # super basic cache (dictionary)
        self.basic_cache = dict()

    def get_nll_musicgen(self, audio, sr):
        inputs = self.processor(
            audio=audio,
            sampling_rate=sr,
            text=None,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            enc = self.model.audio_encoder.encode(inputs["input_values"])
            codes = enc.audio_codes.squeeze(1)  

            decoder_input_ids = codes[..., :-1]
            labels = codes[..., 1:]

            batch_size = codes.shape[0]

            dummy_text_ids = torch.zeros((batch_size, 1), 
                                         dtype=torch.long, device=self.device)
            dummy_text_mask = torch.ones_like(dummy_text_ids)

            outputs = self.model(
                input_ids=dummy_text_ids,            
                attention_mask=dummy_text_mask,      
                decoder_input_ids=decoder_input_ids, 
                labels=labels                        
            )

        return outputs.loss.item()

    @cache
    def get_nll(self, audio_file_path):
        """
        general function to case on model type, currently only supports musicgen
        function should be cached
        """
        if audio_file_path in self.basic_cache: 
            return self.basic_cache[audio_file_path]
        
        if "musicgen" in self.model_name:
            audio, sr = librosa.load(audio_file_path, sr=self.sr)
            return self.get_nll_musicgen(audio, sr)
        else:
            raise Exception("model type not supported -- currently only " \
            "MERT-like models are supported")

if __name__ == "__main__":
    # test it works
    extractor = NLLExtractor()
    sample_file_path = "/Users/lizchu413/Downloads/tv_static_sound_effect_-_bzz.m4a"

    print(f"loading in sample waveform {sample_file_path}...")

    nll = extractor.get_nll(sample_file_path)

    print(f"NLL: {nll:.4f}")
    
