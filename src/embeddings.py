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


def get_word_vector(token: str, model: Word2Vec, embedding_source: str = "input") -> np.ndarray:
    """Return W-only or mean(W, W') vector for one known token."""
    token_index = model.wv.key_to_index[token]
    input_vector = model.wv.vectors[token_index]

    if embedding_source == "input":
        return input_vector

    if embedding_source == "input_output_mean":
        if not hasattr(model, "syn1neg"):
            raise ValueError(
                "input_output_mean requires a Word2Vec model trained with negative sampling."
            )
        output_vector = model.syn1neg[token_index]
        return ((input_vector + output_vector) / 2).astype(np.float32)

    raise ValueError(f"Unsupported Word2Vec embedding_source: {embedding_source}")


def mean_pool_tokens(
    tokens: Iterable[str],
    model: Word2Vec,
    vector_size: int,
    embedding_source: str = "input",
) -> np.ndarray:
    """Average known token vectors into one fixed-size sentence vector."""
    vectors: List[np.ndarray] = [
        get_word_vector(token, model, embedding_source)
        for token in tokens
        if token in model.wv.key_to_index
    ]

    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def texts_to_mean_vectors(
    tokenized_texts: Sequence[Sequence[str]],
    model: Word2Vec,
    vector_size: int,
    embedding_source: str = "input",
) -> np.ndarray:
    """Transform tokenized sentences into a dense matrix for the ANN."""
    return np.vstack(
        [
            mean_pool_tokens(tokens, model, vector_size, embedding_source)
            for tokens in tokenized_texts
        ]
    ).astype(np.float32)
