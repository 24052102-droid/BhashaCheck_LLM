import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)

import torch.nn.functional as F


MODEL_NAME = "google/muril-base-cased"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME
)

model.eval()


THRESHOLD = 0.75


# ============================================
# CALCULATE CONFIDENCE
# ============================================

def calculate_confidence(sentence):

    tokens = tokenizer(
        sentence,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"]

    with torch.no_grad():

        outputs = model(**tokens)

    logits = outputs.logits

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    token_confidences = []

    for i in range(
        1,
        input_ids.shape[1] - 1
    ):

        token_id = input_ids[0, i]

        token_probability = probabilities[
            0,
            i,
            token_id
        ].item()

        token_confidences.append(
            token_probability
        )

    if not token_confidences:
        return 0.5

    confidence = (
        sum(token_confidences) /
        len(token_confidences)
    )

    return confidence


# ============================================
# DETECT FUNCTION
# ============================================

def detect(
    sentence,
    sentence_number,
    ground_truth="",
    language_file=""
):

    confidence = calculate_confidence(sentence)

    status = "CORRECT"

    if confidence < THRESHOLD:

        status = "WRONG"

    # Temporary prediction
    predicted_sentence = sentence

    return {

        "sentence_number": sentence_number,

        "language_file": language_file,

        "original_sentence": sentence,

        "ground_truth": ground_truth,

        "predicted_sentence": predicted_sentence,

        "confidence": confidence,

        "status": status
    }