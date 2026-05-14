import os


# ============================================
# READ DATASET FILES
# ============================================

def read_dataset_files(dataset_folder="dataset"):

    all_sentences = []

    sentence_counter = 1

    for filename in os.listdir(dataset_folder):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(
            dataset_folder,
            filename
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        for line in lines:

            sentence = line.strip()

            if not sentence:
                continue

            # ====================================
            # DATA FORMAT
            # ====================================

            # Currently:
            # sentence itself used as ground truth

            # Later you can replace with:
            # noisy_sentence || corrected_sentence

            ground_truth = sentence

            all_sentences.append({

                "sentence_number":
                    sentence_counter,

                "sentence":
                    sentence,

                "ground_truth":
                    ground_truth,

                "language_file":
                    filename
            })

            sentence_counter += 1

    return all_sentences


# ============================================
# SAVE RESULTS
# ============================================

def save_results(results, output_file):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for result in results:

            f.write(
                f"Sentence "
                f"{result['sentence_number']}\n"
            )

            f.write(
                "{\n"
            )

            f.write(
                f"  'sentence_number': "
                f"{result['sentence_number']},\n"
            )

            f.write(
                f"  'language_file': "
                f"'{result['language_file']}',\n"
            )

            f.write(
                f"  'original_sentence': "
                f"'{result['original_sentence']}',\n"
            )

            f.write(
                f"  'ground_truth': "
                f"'{result['ground_truth']}',\n"
            )

            f.write(
                f"  'predicted_sentence': "
                f"'{result['predicted_sentence']}',\n"
            )

            f.write(
                f"  'confidence': "
                f"{result['confidence']:.4f},\n"
            )

            f.write(
                f"  'status': "
                f"'{result['status']}'\n"
            )

            f.write(
                "}\n\n"
            )


# ============================================
# SAVE WRONG SENTENCE IDS
# ============================================

def save_wrong_sentence_numbers(
    results,
    output_file
):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for result in results:

            if result["status"] == "WRONG":

                f.write(
                    f"Sentence "
                    f"{result['sentence_number']}\n"
                )

                f.write(
                    "{\n"
                )

                f.write(
                    f"  'language_file': "
                    f"'{result['language_file']}',\n"
                )

                f.write(
                    f"  'original_sentence': "
                    f"'{result['original_sentence']}',\n"
                )

                f.write(
                    f"  'ground_truth': "
                    f"'{result['ground_truth']}',\n"
                )

                f.write(
                    f"  'predicted_sentence': "
                    f"'{result['predicted_sentence']}',\n"
                )

                f.write(
                    f"  'confidence': "
                    f"{result['confidence']:.4f},\n"
                )

                f.write(
                    f"  'status': "
                    f"'{result['status']}'\n"
                )

                f.write(
                    "}\n\n"
                )