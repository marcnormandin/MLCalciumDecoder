# MLCalciumDecoder
An animal position decoder, Python 3, using neural activity measured with calcium transients from CA1 using binary classifiers.

Neural activity recorded as calcium transients, in conjunction with positional data, are used to build a model composed of thousdands of binary classifiers for a two-dimensional binned region of space.

The decoder consists of one binary classifier trained for each pair of bins (if the bins meet requirements). When decoding neural activity, each classifier votes for which of two distinct spational bin locations is most likely to be the true location of the animal. The result, after all voting, a two-dimensional map of likely activity.

This has been tested with synthetic neural activity in conjunction with a real animal path, as well as real neural activity from CA1 with a real animal path.

It builds the model based on the first X% of the data, and then predicts the remaining percent, which is data that it hasn't seen.
