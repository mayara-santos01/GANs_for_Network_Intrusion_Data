import csv
from itertools import zip_longest

labels_file = "svmsmote_generated_labels.txt"
samples_file = "svmsmote_decoded_generated_samples.csv"
output_file = "svmsmote_merged.csv"

classes = {
    0: "Analysis",
    1: "Backdoor",
    2: "DoS",
    3: "Exploits",
    4: "Fuzzers",
    5: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms",
}

# Lê os rótulos
with open(labels_file, "r", encoding="utf-8") as f:
    labels = [int(char) for char in f.read() if char.isdigit()]

# Lê as amostras do CSV, preservando os nomes das features
with open(samples_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)

    feature_names = next(reader)

    samples = [
        row for row in reader
        if row and any(value.strip() for value in row)
    ]

if not samples:
    raise ValueError("O arquivo de amostras está vazio.")

num_features = len(feature_names)
rows = []

# Combina amostras e labels
for i, (label, sample) in enumerate(
    zip_longest(labels, samples, fillvalue=None),
    start=1
):
    if sample is None:
        sample = ["NA"] * num_features

    elif len(sample) < num_features:
        sample += ["NA"] * (num_features - len(sample))

    elif len(sample) > num_features:
        raise ValueError(
            f"Número incorreto de features na amostra {i}: "
            f"{len(sample)} (esperado: {num_features})"
        )

    if label is None:
        class_id = "NA"
        class_name = "NA"
    else:
        class_id = label
        class_name = classes.get(class_id)

        if class_name is None:
            raise ValueError(
                f"ID de classe desconhecido na posição {i}: {label}"
            )

    rows.append([*sample, class_id, class_name])

# Preserva os nomes originais das colunas
header = feature_names + ["class_id", "class_name"]

# Exporta o CSV final
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"CSV criado: {output_file}")
print(f"Rótulos: {len(labels)} | Amostras: {len(samples)}")
print(f"Features: {num_features}")
print(f"Total de registros: {len(rows)}")
