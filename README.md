# Natserract AI

```sh
poetry shell

poetry install
```

## How it works
```sh
Training Model -> 
```

## Requirements
- Postgres 15
- Enable the extension
```sql 
CREATE EXTENSION vector;
```

## Running

```sh
poetry run python main.py
```

## Performance Considerations

If you need to perform this operation frequently and especially if the set of word vectors is large, it may be practical to use a database or a data store optimized for vector operations. These data stores can persist your word vectors and provide efficient similarity search functionality:

- FAISS by Facebook AI Research is a library for efficient similarity search and clustering of dense vectors.
- Elasticsearch has plugins like elasticsearch-vector-scoring to handle vector similarity.
- Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point.

Using such systems can significantly speed up the similarity search process.