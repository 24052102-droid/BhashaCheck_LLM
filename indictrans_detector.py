import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.eval()


THRESHOLD = 0.60


# ============================================
# LANGUAGE MAPPING
# ============================================

LANG_FILE_TO_CODE = {

    "hindi.txt": "hin_Deva",

    "kannada.txt": "kan_Knda",

    "telugu.txt": "tel_Telu",

    "bhojpuri.txt": "bho_Deva",

    "maithili.txt": "mai_Deva"
}

DEFAULT_LANG_CODE = "hin_Deva"

TGT_LANG = "eng_Latn"


# ============================================
# CALCULATE CONFIDENCE
# ============================================

def calculate_confidence(
    sentence,
    src_lang
):

    tokenizer.src_lang = src_lang

    tokenizer.tgt_lang = TGT_LANG

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    tgt_lang_id = tokenizer.convert_tokens_to_ids(
        TGT_LANG
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=64,
            forced_bos_token_id=tgt_lang_id
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

    src_lang = LANG_FILE_TO_CODE.get(
        language_file,
        DEFAULT_LANG_CODE
    )

    confidence = calculate_confidence(
        sentence,
        src_lang
    )

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