import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration


MODEL_NAME = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = MT5ForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.eval()


THRESHOLD = 0.65


# ============================================
# CALCULATE CONFIDENCE
# ============================================

def calculate_confidence(sentence):

    input_text = f"detect spelling: {sentence}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=64
        )

    scores = outputs.scores

    if not scores:
        return 0.5

    probabilities = []

    for score_tensor in scores:

        probs = torch.softmax(
            score_tensor,
            dim=-1
        )

        max_prob = probs.max().item()

        probabilities.append(max_prob)

    confidence = (
        sum(probabilities) /
        len(probabilities)
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