# MLCalciumDecoder
An animal position decoder, Python 3, using neural activity measured with calcium transients from CA1 using binary classifiers.

Neural activity recorded as calcium transients, in conjunction with positional data, are used to build a model composed of thousdands of binary classifiers for a two-dimensional binned region of space.

The decoder consists of one binary classifier trained for each pair of bins (if the bins meet requirements). When decoding neural activity, each classifier votes for which of two distinct spatial bin locations is most likely to be the true location of the animal. The result, after all classifiers have voted, is a two-dimensional map of likely position. The predicted location, for the given neural activity at a single timestamp, is the bin with the most votes (but this can be improved). This is repeated for each timestamp of data, resulting in a predicted path of an animal.

This has been tested with synthetic neural activity in conjunction with a real animal path, as well as real neural activity from CA1 with a real animal path.

It builds the model based on the first X% of the data, and then predicts the remaining percent, which is data that it hasn't seen.

The results are saved to a Python pickle file for re-use and analysis.

There is a separate Python notebook that further processes the decoder results.
