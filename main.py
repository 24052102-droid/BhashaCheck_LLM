import os

from utils import (
    read_dataset_files,
    save_results,
    save_wrong_sentence_numbers
)

import muril_detector
import mt5_detector
import indictrans_detector


# ============================================
# CREATE REQUIRED DIRECTORIES
# ============================================

os.makedirs("outputs", exist_ok=True)

os.makedirs("wrong_sentences", exist_ok=True)

os.makedirs("results", exist_ok=True)


# ============================================
# LOAD DATASET
# ============================================

all_sentences = read_dataset_files()

print(
    f"Total sentences loaded: "
    f"{len(all_sentences)}\n"
)


# ============================================
# RESULT STORAGE
# ============================================

muril_results = []

mt5_results = []

indictrans_results = []


# ============================================
# MURIL
# ============================================

print("Running MuRIL...\n")

for item in all_sentences:

    try:

        # sentence
        sentence = item["sentence"]

        # sentence number
        sentence_number = item["sentence_number"]

        # language file
        language_file = item["language_file"]

        # OPTIONAL:
        # if corrected sentence exists in dataset
        ground_truth = item.get(
            "ground_truth",
            sentence
        )

        result = muril_detector.detect(
            sentence=sentence,

            sentence_number=sentence_number,

            ground_truth=ground_truth,

            language_file=language_file
        )

        muril_results.append(result)

        print(result)

    except Exception as e:

        print(
            f"[MuRIL ERROR] "
            f"Sentence {item['sentence_number']}: {e}"
        )

print(
    f"\nMuRIL collected "
    f"{len(muril_results)} results."
)


# ============================================
# MT5
# ============================================

print("\nRunning mT5...\n")

for item in all_sentences:

    try:

        sentence = item["sentence"]

        sentence_number = item["sentence_number"]

        language_file = item["language_file"]

        ground_truth = item.get(
            "ground_truth",
            sentence
        )

        result = mt5_detector.detect(
            sentence=sentence,

            sentence_number=sentence_number,

            ground_truth=ground_truth,

            language_file=language_file
        )

        mt5_results.append(result)

        print(result)

    except Exception as e:

        print(
            f"[mT5 ERROR] "
            f"Sentence {item['sentence_number']}: {e}"
        )

print(
    f"\nmT5 collected "
    f"{len(mt5_results)} results."
)


# ============================================
# INDICTRANS2
# ============================================

print("\nRunning IndicTrans2...\n")

for item in all_sentences:

    try:

        sentence = item["sentence"]

        sentence_number = item["sentence_number"]

        language_file = item["language_file"]

        ground_truth = item.get(
            "ground_truth",
            sentence
        )

        result = indictrans_detector.detect(
            sentence=sentence,

            sentence_number=sentence_number,

            ground_truth=ground_truth,

            language_file=language_file
        )

        indictrans_results.append(result)

        print(result)

    except Exception as e:

        import traceback

        print(
            f"[IndicTrans2 ERROR] "
            f"Sentence {item['sentence_number']}: {e}"
        )

        traceback.print_exc()

print(
    f"\nIndicTrans2 collected "
    f"{len(indictrans_results)} results."
)


# ============================================
# SAVE OUTPUT FILES
# ============================================

save_results(
    muril_results,
    "outputs/muril_results.txt"
)

save_results(
    mt5_results,
    "outputs/mt5_results.txt"
)

save_results(
    indictrans_results,
    "outputs/indictrans_results.txt"
)


# ============================================
# SAVE WRONG SENTENCE IDS
# ============================================

save_wrong_sentence_numbers(
    muril_results,
    "wrong_sentences/muril_wrong.txt"
)

save_wrong_sentence_numbers(
    mt5_results,
    "wrong_sentences/mt5_wrong.txt"
)

save_wrong_sentence_numbers(
    indictrans_results,
    "wrong_sentences/indictrans_wrong.txt"
)


# ============================================
# FINAL MESSAGE
# ============================================

print("\n================================")

print("All outputs saved successfully.")

print("================================")


print("\nGenerated Files:")

print("- outputs/muril_results.txt")

print("- outputs/mt5_results.txt")

print("- outputs/indictrans_results.txt")

print("- wrong_sentences/muril_wrong.txt")

print("- wrong_sentences/mt5_wrong.txt")

print("- wrong_sentences/indictrans_wrong.txt")