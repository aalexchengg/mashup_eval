from matching.cocola.contrastive_model import constants
from matching.cocola.contrastive_model.contrastive_model import CoCola
from feature_extraction.feature_extraction import CoColaFeatureExtractor

print("loading in model")
model = CoCola.load_from_checkpoint("/Users/abcheng/Documents/workspace/mashup_eval/data/cocola_model/checkpoint-epoch=87-val_loss=0.00.ckpt",
                                    input_type=constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE)
print("model loaded.")
feature_extractor = CoColaFeatureExtractor()

model.eval()