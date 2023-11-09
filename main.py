import glob
import os
from collections import namedtuple

import numpy as np
import spacy
from gensim.models import Word2Vec

from gensim.test.utils import common_texts
from utils import load_markdowns, markdown_to_text, get_vector, get_base_path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    return tokens


def create_models():
    datasets = []
    for (markdown, name) in load_markdowns():
        content = markdown_to_text(markdown)
        model = Word2Vec(sentences=content, vector_size=100, window=5, min_count=1, workers=4)
        model.save(f"models/{name.replace('.md', '')}.model")
        datasets.extend(preprocess_text(content))

    return datasets


def load_models() -> list[Word2Vec]:
    path = get_base_path(os.path.join('models'))
    pattern = os.path.join(path, '*.model')

    models_path = []
    for model_path in glob.glob(pattern):
        models_path.append(Word2Vec.load(model_path))

    return models_path


def aggregate_vectors(word2vec_model, document):
    vectors = [word2vec_model.wv[word] for word in document if word in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def main():
    markdown_documents = [content for content, name in load_markdowns()]
    tagged_documents = [
        TaggedDocument(words=preprocess_text(markdown_to_text(md)), tags=[str(i)])
        for i, md in enumerate(markdown_documents)
    ]

    # Train a Doc2Vec model
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

    # Infer vectors for the original documents
    # document_vectors = [model.infer_vector(doc.words) for doc in tagged_documents]

    # Finding
    query_text = 'Haskell'
    query_tokens = preprocess_text(query_text)
    query_vector = model.infer_vector(query_tokens)

    # Compute similarities between the query and all document vectors
    sims = model.dv.most_similar([query_vector], topn=len(model.dv))

    # Now `sims` is a list of tuples (doc_id, similarity) sorted by similarity
    # To retrieve the most similar documents you can do:
    most_similar_docs = [(tagged_documents[int(sim[0])].words, sim[1]) for sim in sims[:5]]
    all_words = [word for words, score in most_similar_docs for word in words]

    for index, (words, score) in enumerate(most_similar_docs):
        print(f"Document {index + 1}:")
        print("Words:", " ".join(words))
        print("Similarity score:", score)
        print("\n")

    document_text = ' '.join(all_words)
    print('document_text', document_text)

main()
