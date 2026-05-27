from typing import Iterable, List, Sequence

import numpy as np
from gensim.models import Word2Vec


def train_word2vec(tokenized_texts: Sequence[Sequence[str]], config: dict) -> Word2Vec:
    """Train a Gensim Word2Vec model from tokenized sentences."""
    return Word2Vec(
        sentences=tokenized_texts,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        workers=config["workers"],
        sg=config["sg"],
        negative=config["negative"],
        seed=config["seed"],
        epochs=config["epochs"],
    )


def mean_pool_tokens(tokens: Iterable[str], model: Word2Vec, vector_size: int) -> np.ndarray:
    """Average known token vectors into one fixed-size sentence vector."""
    vectors: List[np.ndarray] = [
        model.wv[token]
        for token in tokens #mean pooling với ma trận W, tức vector 100 chiều 
        if token in model.wv.key_to_index
    ]

    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def texts_to_mean_vectors(
    tokenized_texts: Sequence[Sequence[str]],
    model: Word2Vec,
    vector_size: int,
) -> np.ndarray:
    """Transform tokenized sentences into a dense matrix for the ANN."""
    return np.vstack(
        [mean_pool_tokens(tokens, model, vector_size) for tokens in tokenized_texts]
    ).astype(np.float32)
