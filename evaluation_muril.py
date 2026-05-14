# ============================================
# MuRIL Spell Checker Evaluation
# ============================================

# pip install pandas matplotlib

import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# FILE PATH
# ============================================

file_path = "../outputs/muril_results.txt"


# ============================================
# VARIABLES
# ============================================

correct = 0
wrong = 0

confidences = []

language_stats = {}


# ============================================
# READ FILE
# ============================================

with open(file_path, "r", encoding="utf-8") as f:

    lines = f.readlines()


# ============================================
# PROCESS LINES
# ============================================

for line in lines:

    match = re.search(
        r"File: (.*?) \| (CORRECT|WRONG) \| Confidence: ([0-9.]+)",
        line
    )

    if match:

        language = match.group(1)

        label = match.group(2)

        confidence = float(match.group(3))

        confidences.append(confidence)

        # Initialize language stats
        if language not in language_stats:

            language_stats[language] = {
                "correct": 0,
                "wrong": 0,
                "confidence": []
            }

        # Count correct/wrong
        if label == "CORRECT":

            correct += 1

            language_stats[language]["correct"] += 1

        else:

            wrong += 1

            language_stats[language]["wrong"] += 1

        language_stats[language]["confidence"].append(
            confidence
        )


# ============================================
# FINAL METRICS
# ============================================

total = correct + wrong

accuracy = correct / total

avg_confidence = sum(confidences) / len(confidences)


# ============================================
# PRINT RESULTS
# ============================================

print("\n==============================")
print("MuRIL Evaluation Results")
print("==============================")

print(f"Total Sentences: {total}")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")

print(f"\nAccuracy: {accuracy:.4f}")

print(f"Average Confidence: {avg_confidence:.4f}")


# ============================================
# LANGUAGE-WISE RESULTS
# ============================================

print("\n==============================")
print("Language-wise Performance")
print("==============================")

summary_rows = []

for language, stats in language_stats.items():

    lang_total = (
        stats["correct"] +
        stats["wrong"]
    )

    lang_accuracy = (
        stats["correct"] / lang_total
    )

    lang_avg_conf = (
        sum(stats["confidence"]) /
        len(stats["confidence"])
    )

    print(f"\nLanguage: {language}")

    print(f"Correct: {stats['correct']}")

    print(f"Wrong: {stats['wrong']}")

    print(f"Accuracy: {lang_accuracy:.4f}")

    print(
        f"Average Confidence: "
        f"{lang_avg_conf:.4f}"
    )

    summary_rows.append({
        "Language": language,
        "Correct": stats["correct"],
        "Wrong": stats["wrong"],
        "Accuracy": round(lang_accuracy, 4),
        "Average Confidence": round(
            lang_avg_conf,
            4
        )
    })


# ============================================
# SAVE CSV
# ============================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    "../results/muril_summary.csv",
    index=False
)

print("\nSaved: muril_summary.csv")


# ============================================
# BAR GRAPH
# ============================================

languages = summary_df["Language"]

accuracies = summary_df["Accuracy"]

plt.figure(figsize=(10, 5))

plt.bar(languages, accuracies)

plt.xlabel("Languages")

plt.ylabel("Accuracy")

plt.title("MuRIL Language-wise Accuracy")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "../results/muril_accuracy_graph.png"
)

print(
    "Saved: muril_accuracy_graph.png"
)

plt.show()