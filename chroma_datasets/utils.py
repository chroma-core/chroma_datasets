
from datasets import load_dataset, DatasetInfo, Dataset, load_from_disk
from typing import Optional, Union, Sequence, Dict, Mapping, List
from chroma_datasets.types import AddEmbedding, Datapoint


def load_huggingface_dataset(dataset_name, split_name=None):
    """
    Loads a dataset from Hugging Face datasets library.

    This is a very minimal wrapper, but we are including it to make it easier to follow the logic
    for importing future kinds of datasets.

    Args:
        dataset_name (str): Name of the dataset to load.
        split_name (str, optional): Name of the split to load (e.g., "train", "test", "validation"). 
                                   If not provided, the entire dataset will be loaded.

    Returns:
        dataset (datasets.Dataset): The loaded dataset.
    """
    dataset = load_dataset(dataset_name, split=split_name)
    return dataset


def transform_data(dataset, mapping):
    """
    Transforms a dataset using a mapping.

    Optional utility function that can be used to transform a dataset into a format that Chroma can understand.

    See `glue.py` for an example
    """
    # Create a new list for the transformed data
    transformed_data = []

    for i, row in enumerate(dataset):
        # Create a new dictionary for the transformed row
        transformed_row = {}

        # Process each item in the mapping
        for key, value in mapping.items():
            # If the value is a string, perform a direct mapping
            if isinstance(value, str):
                transformed_row[key] = row[value]

            # If the value is a function, use it to transform the data
            elif callable(value):
                transformed_row[key] = value(row)

            else:
                raise ValueError(f"Invalid mapping for {key}: {value}")

        # Add the transformed row to the list
        transformed_data.append(transformed_row)

    # Return the transformed data
    return transformed_data


def to_chroma_schema(datapoints: List[Datapoint]) -> AddEmbedding:
    """
    Converts a list of datapoints into the AddEmbedding schema.
    """
    embeddings = []
    metadatas = []
    documents = []
    ids = []

    for datapoint in datapoints:

        # datapoint is an object, check to see if it has a key called "embedding"
        # if it does, add it to the add_embedding object
        # if it doesn't, do nothing
        if "embedding" in datapoint:
            embeddings.append(datapoint["embedding"])
        if "metadata" in datapoint:
            metadatas.append(datapoint["metadata"])
        if "document" in datapoint:
            documents.append(datapoint["document"])
        if "id" in datapoint:
            ids.append(datapoint["id"])

    return {
        "ids": ids if len(ids) > 0 else None,
        "metadatas": metadatas if len(metadatas) > 0 else None,
        "documents": documents if len(documents) > 0 else None,
        "embeddings": embeddings if len(embeddings) > 0 else None,
    }


def import_into_chroma(chroma_client, dataset, collection_name=None, embedding_function=None):
    """
    Imports a dataset into Chroma.

    Args:
        chroma_client (ChromaClient): The ChromaClient to use.
        collection_name (str): The name of the collection to load the dataset into.
        dataset (AddEmbedding): The dataset to load.
        embedding_function (Optional[Callable[[str], np.ndarray]]): A function that takes a string and returns an embedding.
    """
    # if chromadb is not install, raise an error
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        raise ImportError("Please install chromadb to use this function. `pip install chromadb`")

    ef = None

    if dataset.embedding_function is not None:
        if embedding_function is None:
            error_msg = "See documentation"
            if dataset.embedding_function_instructions is not None:
                error_msg = dataset.embedding_function_instructions

            raise ValueError(f"""
                             Dataset requires embedding function: {dataset.embedding_function}.

                             {error_msg}
                             """)

        if embedding_function.__class__.__name__ != dataset.embedding_function:
            raise ValueError(f"Please use {dataset.embedding_function} as the embedding function for this dataset. You passed {embedding_function.__class__.__name__}")

    if embedding_function is not None:
        ef = embedding_function

    # if collection_name is None, get the name from the dataset type 
    if collection_name is None:
        collection_name = dataset.__name__

    if ef is None:
        ef = embedding_functions.DefaultEmbeddingFunction()

    collection = chroma_client.create_collection(
        collection_name,
        embedding_function=ef
        )

    mapped_data = dataset.to_chroma()

    collection.add(
        ids=mapped_data["ids"],
        metadatas=mapped_data["metadatas"],
        documents=mapped_data["documents"],
        embeddings=mapped_data["embeddings"],
    )

    print(f"Loaded {len(mapped_data['ids'])} documents into the collection named: {collection_name}")

    return collection


def import_chroma_exported_hf_dataset_from_disk(chroma_client, path, collection_name, embedding_function=None):
    dataset = load_from_disk(path)
    return import_chroma_exported_hf_dataset(chroma_client, dataset, collection_name, embedding_function=embedding_function)


def import_chroma_exported_hf_dataset(chroma_client, dataset, collection_name, embedding_function=None):
    """
        Imports a dataset that was exported from Chroma into a Hugging Face dataset.
        This will only work with data exported from Chroma and is not a generic utility function.
    """
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        raise ImportError("Please install chromadb to use this function. `pip install chromadb`")

    if embedding_function is None:
        print("Caution: No embedding function provided. Using the default embedding function.")
        embedding_function = embedding_functions.DefaultEmbeddingFunction()

    if collection_name is None:
        raise ValueError("Please provide a collection name.")

    # dataset = load_from_disk(path)
    mapped_data = to_chroma_schema(dataset)

    collection = chroma_client.create_collection(
        collection_name,
        embedding_function=embedding_function
        )

    collection.add(
        ids=mapped_data["ids"],
        metadatas=mapped_data["metadatas"],
        documents=mapped_data["documents"],
        embeddings=mapped_data["embeddings"],
    )

    print(f"Loaded {len(mapped_data['ids'])} documents into the collection named: {collection_name}")

    return collection


def export_collection_to_hf_dataset_to_disk(chroma_client, collection_name, path, license="MIT"):
    dataset = export_collection_to_hf_dataset(chroma_client, collection_name, license=license)
    dataset.save_to_disk(path)
    print(f"Saved dataset to {path}")


def export_collection_to_hf_dataset(chroma_client, collection_name, license="MIT"):
    """
    Exports a Chroma collection to a Hugging Face dataset.

    Args:
        chroma_client (ChromaClient): The ChromaClient to use.
        collection_name (str): The name of the collection to export.
        license (str): The license to use for the dataset. Default MIT
    """
    collection = chroma_client.get_collection(collection_name)

    # add pagination later
    all_data = collection.get(include=["documents", "embeddings", "metadatas"])
    
    row_data = []
    for id, embdding, metadata, document in zip(all_data["ids"], all_data["embeddings"], all_data["metadatas"], all_data["documents"]):
        row_data.append({"id": id, "embedding": embdding, "metadata": metadata, "document": document})

    dataset_info = DatasetInfo(
            license=license,
        )

    dataset = Dataset.from_list(
        row_data,
        info=dataset_info
    )

    return dataset