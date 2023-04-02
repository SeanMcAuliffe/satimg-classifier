# Counts the proportion of positive and negative labels in the binary dataset
import numpy as np

labels = []
with open('../data/labels/labels_binary_minerals_reg.csv', 'r') as f:
    for line in f.readlines():
        name, label = line.strip().split(',')
        labels.append(label)

print("Largest label: ", max(labels))
print("10th largest label: ", sorted(labels)[-10])
print("10th percentile label: ", np.percentile(labels, 10))
print("25th percentile label: ", np.percentile(labels, 25))
print("50th percentile label: ", np.percentile(labels, 50))
print("75th percentile label: ", np.percentile(labels, 75))
print("90th percentile label: ", np.percentile(labels, 90))
print("99th percentile label: ", np.percentile(labels, 99))