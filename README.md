### 🔍
## Chroma Datasets 

```
pip install chroma_datasets
```

- a public package registry of **sample and useful datasets** to use with embeddings
- a set of tools to **export and import** Chroma collections

We built to enable **faster experimentation**: There is no good source of sample datasets and sample datasets are incredibly important to enable fast experiments and learning.



### Current Datasets
| Dataset                | Size              | Contributor            | Python Class                   |
|------------------------|-------------------|------------------------|--------------------------------|
| State of the Union     | 51kb               | Chroma        | `from chroma_datasets import StateOfTheUnion` |
| Paul Graham Essay      | 1.3mb               | Chroma        | `from chroma_datasets import PaulGrahamEssay` |
| Huberman Podcasts | 4.3mb | [Dexa AI](https://dexa.ai/) | `from chroma_datasets import HubermanPodcasts` 
| more soon... | | | read below how to contribute
 
`chroma_datasets` is currently backed by Hugging Face datasets

### How to use

The following will:
1. Download the 2022 State of the Union
2. Chunk it up for you
3. Embed it using Chroma's default open-source embedding function
4. Import it into Chroma

Try it yourself in this [Colab Notebook](https://githubtocolab.com/chroma-core/chroma_datasets/blob/master/examples/super_simple.ipynb).

```python
import chromadb
from chromadb.utils import embedding_functions
from chroma_datasets import StateOfTheUnion
from chroma_datasets.utils import import_into_chroma

chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="API_KEY",
    model_name="text-embedding-ada-002"
)
collection = import_into_chroma(chroma_client=chroma_client, dataset=StateOfTheUnion, embedding_function=openai_ef)
result = collection.query(query_texts=["The United States of America"], n_results=1)
print(result)
```

### Adding a New Dataset

We welcome new datasets! 

These datasets can be anything generally useful to developer education for processing and using embeddings. 

Datasets should be exported from a Chroma collection. See `examples/example_export.ipynb` for an example of how to create a dataset on Hugging Face (the default path)

#### Create a new dataset from a Chroma Collection

(more examples of this in `examples/example_export.ipynb`)

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
```python
class PaulGrahamEssay(Dataset):
    """
        http://www.paulgraham.com/worked.html
    """
    hf_data = None
    hf_dataset_name = "chromadb/pg_essay"
    embedding_function = "OpenAIEmbeddingFunction"
    embedding_function_instructions = ef_instruction_dict[embedding_function]
```

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

### License
Code: Apache 2.0

Each dataset has it's own license. Datasets uploaded by Chroma are released as `MIT`.

### Todo

- [ ] Add test suite to test some of the critical paths
- [ ] Add automated pypi release
- [ ] Add loaders for other locations (remote like S3, local like CSV... etc)
- [ ] Super easy streamlit/gradio wrapper to push up a collection to interact with
