from coordinators.documents.create import create_tagged_documents_by_document_id
from coordinators.models.read import retrieve_models
from database import connect
from helpers.vectorize import preprocess_text


async def count_documents():
    try:
        conn = await connect()
        results = await conn.fetchval(
            f"select count(*) from documents")

        return results
    except Exception as e:
        raise ValueError("Can't get documents")


async def get_all():
    try:
        conn = await connect()
        results = await conn.fetch(
            f"select * from documents docs")

        return results
    except Exception as e:
        raise ValueError('Documents not found!')


async def get(id: int):
    try:
        conn = await connect()
        result = await conn.fetchrow(
            f"select * from documents doc where doc.id = {id}")

        return result
    except Exception as e:
        raise ValueError('Document not found!')


async def filter_by_similarity_score(
        nlp,
        query: str,
        top_k=5
):
    try:
        documents = await get_all()
        tagged_documents = create_tagged_documents_by_document_id(nlp, documents)

        query_tokens = preprocess_text(nlp(query))
        models = await retrieve_models()

        most_similar_docs = []
        for model in models:
            query_vector = model.infer_vector(query_tokens)

            sims = model.dv.most_similar([query_vector], topn=len(model.dv))
            for tag, similarity in sims[:top_k]:
                sim_idx = int(tag[0])
                sim_document_id = int(tag[1])

                similar_doc = tagged_documents[sim_idx]
                most_similar_docs.append(
                    (
                        sim_document_id,
                        similar_doc.words,
                        similarity
                    )
                )

        return most_similar_docs
    except Exception as e:
        raise ValueError(str(e))
