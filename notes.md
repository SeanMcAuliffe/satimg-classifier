A full dataset has been acquired, it contains 32,371 images and metadata files.
This dataset should be pre-padded to exactly 512x512 using the normalize_dimension.py
script. 

The full dataset has been found to have the following proportion of + / - labels:

Positive: 19819, Negative: 12554, Total: 32372
Positive: 61.22%, Negative: 39.78%

To prevent the model from learning a simple rule like "just usually predict positive
labels", when the dataset is loaded into training and testing sets, it should
be ensured that the training set contains roughly 50% positive and 50% negative
examples. This is implemented in the utils.load_data.load_dataset() function.

Given a training proportion of 80%, the largest experiment which can therefore 
be run on this data is limited by the number of negative examples.

The largest experiment would have the following set sizes:

Training: 10,043 negative examples, 10,043 positive examples, total 20,086
Test Set, 2,511 negative examples, 9,776 positive examples, total 12,287

To run an experiment this large would require overcoming GPU memory limitations,
perhaps by streaming the data from disc. This has not been implemented yet.

As a practical first measure, smaller experiments which maintain the 50/50 split
can be run.

Total number of buckets before removing: 5507               / 6509
Total number of buckets marked for removal: 53              / 1920 # from which images where removed to make it mono-label
Total Images Removed: 108                                    / 9307




Loading brightness values ...
Total unique location buckets: 6509
Removing images over ocean from dataset ...
Marked 1920 buckets for removal
Total 38330 images before
Marked 9307 images for removal
Total after removing ocean buckets: 4589
Total Neg. Buckets: 1564
Total Pos. Buckets: 3025


38330 = Number of images loaded into buckets


Ocean Mask csv Count: 

Positive: 29072, Negative: 9397, Total: 38468
Positive: 0.7557, Negative: 0.2443

Number of images loaded: 
32371
Loading label data ...
Loading coords ...
Loading brightness values ...
Total unique location buckets: 5507
Removing images over ocean from dataset ...
Marked 1816 buckets for removal
Total 32263 iamges before
Marked 8809 iamges for removal
Total after removing ocean buckets: 3691
Total Neg. Buckets: 1171
Total Pos. Buckets: 2520
Sampling training set ...



# -------------------# 
Yesterday's Dataset: 
Number of images loaded: 32,371
Total unique location buckets before mono-labelling: 5507
Total unique location buckets after mono-labelling: 5507
Total 32,263 images after mono-labelling --- removed 108 images
Removing images over ocean from dataset ... Marked 1816 buckets for removal
Marked 8809 iamges for removal
Total after removing ocean buckets: 3691
Total Neg. Buckets: 1171
Total Pos. Buckets: 2520

