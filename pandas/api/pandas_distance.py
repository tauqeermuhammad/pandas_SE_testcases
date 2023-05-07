import pandas as pd
import numpy as np

def euclidean_distance(X: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    """
    Calculate the Euclidean distance between two Pandas DataFrames.
    
    Parameters
    ----------
    X : pd.DataFrame
        The first input DataFrame.
    Y : pd.DataFrame
        The second input DataFrame.
        
    Returns
    -------
    pd.Series
        A Pandas Series with the Euclidean distance between each row of the input DataFrames.
    """
    assert X.shape[1] == Y.shape[1], "Input DataFrames must have the same number of columns."
    distance = np.sqrt(np.sum((X.values[:, None, :] - Y.values[None, :, :]) ** 2, axis=-1))
    return pd.Series(distance.ravel(), index=X.index)

def cosine_similarity(X: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    """
    Calculate the cosine similarity between two Pandas DataFrames.
    
    Parameters
    ----------
    X : pd.DataFrame
        The first input DataFrame.
    Y : pd.DataFrame
        The second input DataFrame.
        
    Returns
    -------
    pd.Series
        A Pandas Series with the cosine similarity between each row of the input DataFrames.
    """
    assert X.shape[1] == Y.shape[1], "Input DataFrames must have the same number of columns."
    norm_X = np.sqrt(np.sum(X.values ** 2, axis=1))
    norm_Y = np.sqrt(np.sum(Y.values ** 2, axis=1))
    similarity = np.dot(X.values, Y.values.T) / np.outer(norm_X, norm_Y)
    return pd.Series(similarity.ravel(), index=X.index)
