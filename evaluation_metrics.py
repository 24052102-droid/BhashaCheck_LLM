# ============================================
# SPELL CHECKER EVALUATION
# CER, WER, MER, ACCURACY
# ============================================

# pip install jiwer pandas matplotlib

import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from jiwer import cer, wer, mer


# ============================================
# RESULT FILES
# ============================================

MODEL_FILES = {

    "MuRIL":
        "../outputs/muril_results.txt",

    "MT5":
        "../outputs/mt5_results.txt",

    "IndicTrans2":
        "../outputs/indictrans_results.txt"
}


# ============================================
# OUTPUT DIRECTORY
# ============================================

os.makedirs("../results", exist_ok=True)


# ============================================
# EVALUATE MODEL
# ============================================

def evaluate_model(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        content = f.read()

    blocks = content.split("Sentence ")

    total_sentences = 0

    correct = 0
    wrong = 0

    total_cer = 0
    total_wer = 0
    total_mer = 0

    total_confidence = 0

    sentence_results = []

    for block in blocks:

        if "ground_truth" not in block:
            continue

        # ====================================
        # EXTRACT FIELDS
        # ====================================

        gt_match = re.search(
            r"'ground_truth': '(.*?)'",
            block,
            re.DOTALL
        )

        pred_match = re.search(
            r"'predicted_sentence': '(.*?)'",
            block,
            re.DOTALL
        )

        conf_match = re.search(
            r"'confidence': ([0-9.]+)",
            block
        )

        status_match = re.search(
            r"'status': '(CORRECT|WRONG)'",
            block
        )

        lang_match = re.search(
            r"'language_file': '(.*?)'",
            block
        )

        if not (
            gt_match and
            pred_match and
            conf_match and
            status_match
        ):
            continue

        # ====================================
        # VALUES
        # ====================================

        ground_truth = gt_match.group(1)

        predicted = pred_match.group(1)

        confidence = float(
            conf_match.group(1)
        )

        status = status_match.group(1)

        language = (
            lang_match.group(1)
            if lang_match else "unknown"
        )

        # ====================================
        # METRICS
        # ====================================

        cer_score = cer(
            ground_truth,
            predicted
        )

        wer_score = wer(
            ground_truth,
            predicted
        )

        mer_score = mer(
            ground_truth,
            predicted
        )

        total_cer += cer_score

        total_wer += wer_score

        total_mer += mer_score

        total_confidence += confidence

        total_sentences += 1

        if status == "CORRECT":
            correct += 1
        else:
            wrong += 1

        sentence_results.append({

            "Language": language,

            "Ground Truth": ground_truth,

            "Predicted": predicted,

            "CER": round(cer_score, 4),

            "WER": round(wer_score, 4),

            "MER": round(mer_score, 4),

            "Confidence": round(
                confidence,
                4
            ),

            "Status": status
        })

    # ========================================
    # FINAL METRICS
    # ========================================

    accuracy = correct / total_sentences

    avg_confidence = (
        total_confidence /
        total_sentences
    )

    avg_cer = (
        total_cer /
        total_sentences
    )

    avg_wer = (
        total_wer /
        total_sentences
    )

    avg_mer = (
        total_mer /
        total_sentences
    )

    return {

        "Accuracy":
            round(accuracy, 4),

        "CER":
            round(avg_cer, 4),

        "WER":
            round(avg_wer, 4),

        "MER":
            round(avg_mer, 4),

        "Average Confidence":
            round(avg_confidence, 4)

    }, sentence_results


# ============================================
# RUN ALL MODELS
# ============================================

summary_rows = []

for model_name, file_path in MODEL_FILES.items():

    print("\n================================")

    print(f"Evaluating {model_name}")

    print("================================")

    metrics, sentence_results = evaluate_model(
        file_path
    )

    print(metrics)

    # ========================================
    # SAVE SENTENCE-LEVEL CSV
    # ========================================

    sentence_df = pd.DataFrame(
        sentence_results
    )

    sentence_csv = (
        f"../results/"
        f"{model_name.lower()}_sentence_metrics.csv"
    )

    sentence_df.to_csv(
        sentence_csv,
        index=False
    )

    print(f"Saved: {sentence_csv}")

    # ========================================
    # SUMMARY
    # ========================================

    summary_row = {

        "Model": model_name,

        **metrics
    }

    summary_rows.append(summary_row)


# ============================================
# SAVE FINAL COMPARISON CSV
# ============================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    "../results/model_comparison.csv",
    index=False
)

print("\n================================")

print("FINAL MODEL COMPARISON")

print("================================")

print(summary_df)

print("\nSaved: ../results/model_comparison.csv")


# ============================================
# PLOT CER GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.bar(
    summary_df["Model"],
    summary_df["CER"]
)

plt.xlabel("Model")

plt.ylabel("CER")

plt.title("Character Error Rate Comparison")

plt.tight_layout()

plt.savefig(
    "../results/cer_comparison.png"
)

print("Saved: cer_comparison.png")


# ============================================
# PLOT WER GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.bar(
    summary_df["Model"],
    summary_df["WER"]
)

plt.xlabel("Model")

plt.ylabel("WER")

plt.title("Word Error Rate Comparison")

plt.tight_layout()

plt.savefig(
    "../results/wer_comparison.png"
)

print("Saved: wer_comparison.png")


# ============================================
# PLOT MER GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.bar(
    summary_df["Model"],
    summary_df["MER"]
)

plt.xlabel("Model")

plt.ylabel("MER")

plt.title("Match Error Rate Comparison")

plt.tight_layout()

plt.savefig(
    "../results/mer_comparison.png"
)

print("Saved: mer_comparison.png")


print("\nEvaluation Complete.")