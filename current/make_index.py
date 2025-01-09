import faiss
import numpy as np

def make_index(data, lang, sent_id):
    """Compiles and writes the FAISS index for the token embeddings of a particular sentence."""
    lang_data = (data[(lang, sent_id)][0][1:-1], data[(lang, sent_id)][1][1:-1])
    embeddings = lang_data[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.float32(embeddings))
    faiss.write_index(index, f'indices/{lang}_{sent_id}')