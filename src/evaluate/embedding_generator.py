# Author: @echu2
# General class to get embeddings of audio file from model
# Currently this class assumes we are in a MERT-like model space, but can probably

import librosa
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class EmbeddingGenerator:
    def __init__(self, model_name="m-a-p/MERT-v1-95M"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.processor_sr = self.processor.sampling_rate

    def convert_mert(self, input_audio):
        """
        converts input audio into model's embedding space
        if using MERT-v1-95M, output should be [13, time steps, 768]
        """
        inputs = self.processor(input_audio, sampling_rate=self.processor_sr, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        return torch.stack(outputs.hidden_states).squeeze()

    def convert(self, input_audio):
        """
        general function to case on model type, currently only supports mert
        """
        if "MERT" in self.model_name:
            # assuming that last layer is the embedding space we want to be in
            return self.convert_mert(input_audio)[-1, ...]
        else:
            raise Exception("model type not supported -- currently only MERT-like models are supported")


if __name__ == "__main__":
    # test it works
    generator = EmbeddingGenerator()
    # load in sample audio
    sample_file_path = "eval/holdout_set/wav_files/000200.wav"

    print(f"loading in sample waveform {sample_file_path}...")
    sample_waveform, _ = librosa.load(sample_file_path, sr=generator.processor_sr)

    print("loading complete. getting embeddings...")

    embeddings = generator.convert(sample_waveform)
    print(f"embeddings exist! shape is {embeddings.shape}")

