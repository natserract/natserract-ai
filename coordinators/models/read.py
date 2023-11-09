import pickle
import tempfile
import time

from gensim.models import Doc2Vec
from scipy import spatial

from coordinators.documents.read import get_all as get_all_documents
from coordinators.documents.create import create_tagged_documents
from database import connect
from helpers.vectorize import preprocess_text


async def count_models():
    try:
        conn = await connect()
        results = await conn.fetchval(
            f"select count(*) from models")

        return results
    except Exception as e:
        raise ValueError("Can't get models")


async def get_all():
    try:
        conn = await connect()
        results = await conn.fetch(
            f"select * from models")

        return results
    except Exception as e:
        raise ValueError("Can't get models")


async def retrieve_models() -> list[Doc2Vec]:
    try:
        models = await get_all()
        if not len(models):
            raise ValueError('Model not found!')

        doc2_vec_models = []
        for model in models:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pickle.loads(model['data']))
                temp_file_path = tmp_file.name

            doc2_vec_model = Doc2Vec.load(temp_file_path)
            doc2_vec_models.append(doc2_vec_model)

        return doc2_vec_models

    except Exception as e:
        raise ValueError(str(e))


def cosine_similarity(vec_a, vec_b):
    return 1 - spatial.distance.cosine(vec_a, vec_b)


def get_words_from_similarities(most_similar_docs):
    all_words = [word for words, score in most_similar_docs for word in words]
    document_text = ' '.join(all_words)
    return document_text


async def filter_by_similarity_score(
        nlp,
        query: str,
        top_k=5
):
    try:
        documents = await get_all_documents()
        contents = [document['content'] for document in documents]
        tagged_documents = create_tagged_documents(nlp, contents)

        query_tokens = preprocess_text(nlp(query))
        models = await retrieve_models()

        most_similar_docs = []
        for model in models:
            query_vector = model.infer_vector(query_tokens)

            sims = model.dv.most_similar([query_vector], topn=len(model.dv))
            most_similar_docs.extend(
                [(tagged_documents[int(sim[0])].words, sim[1]) for sim in sims[:top_k]]
            )

        return most_similar_docs
    except Exception as e:
        raise ValueError(str(e))
