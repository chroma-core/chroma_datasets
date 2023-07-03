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

    def __init__(self):
        self.hf_data = load_huggingface_dataset(
            "chromadb/paul_graham_essay",
            split_name="data"
        )

    def raw_text(self) -> str:
        return "\n".join(self.hf_data["document"])
    
    def chunked(self) -> List[Datapoint]:
        return self.hf_data
    
    def to_chroma(self) -> AddEmbedding:
        return to_chroma_schema(self.chunked())
```

Add it to the manifest at `chroma_datasets/__init__.py` to make it easy for people to retrieve (optional)

### Todo

- [ ] Add test suite to test some of the critical paths
- [ ] Add automated pypi release
- [ ] Add loaders for other locations (remote like S3, local like CSV... etc)
- [ ] Super easy streamlit/gradio wrapper to push up a collection to interact with