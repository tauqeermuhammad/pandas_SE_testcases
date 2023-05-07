import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def count_words(df, column):
    """
    Count the number of words in a given column of text data.
    :param df: A Pandas DataFrame.
    :param column: The name of the column containing text data.
    :return: A new DataFrame with a new column 'word_count'.
    """
    nltk.download('punkt')
    df['word_count'] = df[column].apply(lambda x: len(nltk.word_tokenize(x)))
    return df

def remove_stopwords(df, column):
    """
    Removes stopwords from a given column of text data.
    :param df: A Pandas DataFrame.
    :param column: The name of the column containing text data.
    :return: A new DataFrame with a new column 'text_without_stopwords'.
    """
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    df['text_without_stopwords'] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    return df

def calculate_similarity(df, column1, column2):
    """
    Calculates the similarity between two columns of text data using cosine similarity.
    :param df: A Pandas DataFrame.
    :param column1: The name of the first column containing text data.
    :param column2: The name of the second column containing text data.
    :return: A new DataFrame with a new column 'similarity_score'.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column1] + ' ' + df[column2])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    df['similarity_score'] = similarity_matrix.diagonal(offset=len(df))
    return df
