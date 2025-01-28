import faiss # type: ignore
import numpy as np

def make_index(data, lang, sent_id):
    """Compiles and writes the FAISS index for the token embeddings of a particular sentence."""

    embeddings = data[(lang, sent_id)][1:-1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    faiss.write_index(index, f'indices/{lang}_{sent_id}')

def get_index(data, lang, sent_id):
    embeddings = data[(lang, sent_id)][1:-1]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    return index