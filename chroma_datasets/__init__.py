__version__ = "0.1.1"


def transform_data(dataset, mapping):
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


def to_colmnuar(transformed_data):
    columnar_data = {}

    for row in transformed_data:
        for key, value in row.items():
            if key not in columnar_data:
                columnar_data[key] = []

            columnar_data[key].append(value)

    return columnar_data


def import_dataset(chroma_client, collection_name, dataset, mapping_function, dataset_type: str):

    if dataset_type == "HuggingFaceDataset":

        transformed_data = to_colmnuar(transform_data(dataset, mapping_function))

        collection = chroma_client.create_collection(collection_name)

        # validate that transformed_data has the correct keys
        for key in ["ids", "metadatas", "documents", "embeddings"]:
            if key not in transformed_data:
                transformed_data[key] = None

        collection.add(
            ids=transformed_data["ids"],
            metadatas=transformed_data["metadatas"],
            documents=transformed_data["documents"],
            embeddings=transformed_data["embeddings"],
        )

        print(f"Loaded {len(transformed_data['ids'])} documents into {collection_name}")

    else:
        raise ValueError(f"Type not implemented yet: {type}")