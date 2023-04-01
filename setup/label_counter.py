# Counts the proportion of positive and negative labels in the binary dataset

pos = 0
neg = 0
total = -1

with open('./data/labels/local_ocean_mask.csv', 'r') as f:
    for line in f.readlines():
        name, label = line.strip().split(',')
        total += 1
        if label == '1':
            pos += 1
        else:
            neg += 1

print(f"Positive: {pos}, Negative: {neg}, Total: {total}")
print(f"Positive: {pos/total:.4f}, Negative: {neg/total:.4f}")