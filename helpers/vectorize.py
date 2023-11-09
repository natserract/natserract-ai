import numpy as np


def preprocess_text(doc):
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    return tokens


def aggregate_vectors(word2vec_model, document):
    vectors = [word2vec_model.wv[word] for word in document if word in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)