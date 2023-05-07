import pandas as pd
import numpy as np
import pandas.testing as pdt
from pandas_distance import euclidean_distance, cosine_similarity

# Test case for euclidean_distance
def test_euclidean_distance():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    Y = pd.DataFrame({'a': [2, 3, 4], 'b': [7, 8, 9]})
    expected_result = pd.Series([3.741657, 3.741657, 4.242641], index=X.index)
    result = euclidean_distance(X, Y)
    pdt.assert_series_equal(result.round(6), expected_result, check_names=False)

# Test case for cosine_similarity
def test_cosine_similarity():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    Y = pd.DataFrame({'a': [2, 3, 4], 'b': [7, 8, 9]})
    expected_result = pd.Series([0.992277, 0.997054, 0.994937], index=X.index)
    result = cosine_similarity(X, Y)
    pdt.assert_series_equal(result.round(6), expected_result, check_names=False)

# Run the test cases
test_euclidean_distance()
test_cosine_similarity()
