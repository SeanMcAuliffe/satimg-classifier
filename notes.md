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

Total number of buckets: 5507
Total adjusted: 53 # from which images where removed to make it mono-label
Total Images Removed: 108