import logging
import os
import pickle
import tempfile

from gensim.models.doc2vec import Doc2Vec

from coordinators.documents.create import create as create_documents, create_tagged_documents_by_document_id
from coordinators.documents.read import get_all as get_all_documents
from database import get_session
from helpers.hash import get_model_name_from_content
from models import Models


async def init_doc2_vec_models(nlp):
    try:
        logging.info('Try to init document models')
        async with get_session() as session:
            documents = await get_all_documents()
            if not len(documents):
                documents = await create_documents()

            # Preprocessing markdown content
            contents = [document['content'] for document in documents]
            tagged_documents = create_tagged_documents_by_document_id(nlp, documents)
            doc2vec_model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
            doc2vec_model.build_vocab(tagged_documents)
            doc2vec_model.train(tagged_documents, total_examples=doc2vec_model.corpus_count,
                                epochs=doc2vec_model.epochs)

            # Use the first document's content to generate a model name
            model_name = get_model_name_from_content(contents[0]) + '.model'

            with tempfile.TemporaryDirectory() as temp_dir:
                # Path to save the Doc2Vec model
                model_path = os.path.join(temp_dir, 'doc2vec.model')

                # Save the model
                doc2vec_model.save(model_path)

                # Read the model binary data
                with open(model_path, 'rb') as model_file:
                    model_data = model_file.read()

                # Serialize the model data using pickle
                serialized_model = pickle.dumps(model_data)
                model = Models(
                    name=model_name,
                    data=serialized_model
                )
                session.add(model)

            await session.commit()
            logging.info('All models successfully created')

    except Exception as e:
        print('Error', e)
        raise e
