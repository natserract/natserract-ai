import pickle
import tempfile

from gensim.models import Doc2Vec
from scipy import spatial

from database import connect


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


