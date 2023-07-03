## Chroma Datasets

Making it easy to load data into Chroma since 2023

```
pip install chroma_datasets
```

### Current Datasets
- State of the Union `from chroma_datasets import StateOfTheUnion`
- Paul Graham Essay `from chroma_datasets import PaulGrahamEssay`
- Glue `from chroma_datasets import Glue`
- SciPy `from chroma_datasets import SciPy`

`chroma_datasets` is generally backed by hugging face datasets, but it is not a requirement.

### How to use

The following will:
1. Download the 2022 State of the Union
2. Chunk it up for you
3. Embed it using Chroma's default open-source embedding function
4. Import it into Chroma

```python
import chromadb
from chroma_datasets import StateOfTheUnion
from chroma_datasets.utils import import_into_chroma

chroma_client = chromadb.Client()
collection = import_into_chroma(chroma_client=chroma_client, dataset=StateOfTheUnion)
result = collection.query(query_texts=["The United States of America"])
print(result)
```

### Adding a New Dataset

We welcome new datasets! 

These datasets can be anything generally useful to developer education for processing and using embeddings.

Datasets can be:
- raw text (like `StateOfTheUnion`)
- pre-chunked data (like `SciPy`)
- chunked and embedded (like `PaulGrahamEssay`)

See the examples/upload.ipynb for an example of how to create a dataset on Hugging Face (the default path)

#### Create a new dataset from a Chroma Collection

(more examples of this in `examples/upload.ipynb` and `examples/upload_embeddings.ipynb`)

Install dependencies
```sh
pip install datasets huggingface_hub chromadb
```

Login into Hugging Face
```sh
huggingface-cli login
```

Upload an existing collection to Hugging Face
** Hugging Face requires the data to have a "split name" - I suggest using a default of "data" **
```python
import chromadb
from chroma_datasets.utils import export_collection_to_hf_dataset
client = chromadb.PersistantClient(path="./chroma_data")
dataset = export_collection_to_hf_dataset(
    client=client, 
    collection_name="paul_graham_essay", 
    license="MIT")
dataset.push_to_hub(
    repo_id="chromadb/paul_graham_essay", 
    split="data")
```

Create a Dataset Class
- Set the string name of the embedding function you used to embed the data, this will make it possible for users to use the embeddings. Please also customize the helpful error message so if users pass in no embedding function or the wrong one, they get help.
- `raw_text` is optional
- `chunked` and `to_chroma` you can copy letter for letter if you uploaded with the method above.
```python
class PaulGrahamEssay(Dataset):
    """
    http://www.paulgraham.com/worked.html

    Columns:
        - id: unique identifier for each chunk
        - document: the text of the chunk
        - embedding: the embedding of the chunk (OpenAI-ada-002)
        - metadata: metadata about the chunk
    """
    hf_data = None
    embedding_function = "OpenAIEmbeddingFunction" # name of embedding function inside Chroma
    embedding_function_instructions = """
        from chromadb.utils import embedding_functions
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key="YOUR_API_KEY",
            model_name="text-embedding-ada-002"
        )
    """

    @classmethod
    def load_data(cls):
        cls.hf_data = load_huggingface_dataset(
            "chromadb/paul_graham_essay",
            split_name="data"
        )

    @classmethod
    def raw_text(cls) -> str:
        if cls.hf_data is None:
            cls.load_data()
        return "\n".join(cls.hf_data["document"])
    
    @classmethod
    def chunked(cls) -> List[Datapoint]:
        if cls.hf_data is None:
            cls.load_data()
        return cls.hf_data
    
    @classmethod
    def to_chroma(cls) -> AddEmbedding:
        return to_chroma_schema(cls.chunked())

```

Add it to the manifest at `chroma_datasets/__init__.py` to make it easy for people to retrieve (optional)

### Utility API Documentation

Many of these methods are purely conveneient. This makes it easy to save and load Chroma Collections to disk. See `./examples/example_export.ipynb` for example use.

```python
from chromadb.utils import (
    export_collection_to_hf_dataset,
    export_collection_to_hf_dataset_to_disk,
    import_chroma_exported_hf_dataset_from_disk,
    import_chroma_exported_hf_dataset
)

# Exports a Chroma collection to an in-memory HuggingFace Dataset
def export_collection_to_hf_dataset(chroma_client, collection_name, license="MIT"):

# Exports a Chroma collection to a HF dataset and saves to the path
def export_collection_to_hf_dataset_to_disk(chroma_client, collection_name, path, license="MIT"):

# Imports a HuggingFace Dataset into a Chroma Collection
def import_chroma_exported_hf_dataset(chroma_client, dataset, collection_name, embedding_function=None):

# Imports a HuggingFace Dataset from Disk and loads it into a Chroma Collection
def import_chroma_exported_hf_dataset_from_disk(chroma_client, path, collection_name, embedding_function=None):

```

### Todo

- [ ] Add test suite to test some of the critical paths
- [ ] Add automated pypi release
- [ ] Add loaders for other locations (remote like S3, local like CSV... etc)
- [ ] Super easy streamlit/gradio wrapper to push up a collection to interact with