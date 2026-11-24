"""
Pretrained speech model wrapper for fine-tuning on emotion classification.

Uses HuggingFace's Wav2Vec2ForSequenceClassification, which already includes
mean-pooling over the transformer's hidden states plus a classification head
— we just swap in our own num_labels and (by default) freeze the
convolutional feature encoder, since that part was pretrained on ~1000x
more audio than we have and rarely benefits from further fine-tuning on a
small downstream dataset; only the transformer layers + classifier head
need to adapt to emotion recognition.
"""
from transformers import Wav2Vec2ForSequenceClassification


def build_wav2vec2_model(num_labels, pretrained_name="facebook/wav2vec2-base",
                          freeze_feature_encoder=True):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        use_safetensors=True,  # avoids torch.load entirely — newer transformers
                                 # blocks the .bin/torch.load path on torch < 2.6
                                 # (CVE-2025-32434); safetensors sidesteps it and
                                 # is the safer format regardless.
    )
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    return model
