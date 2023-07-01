from chroma_datasets.types import AddEmbedding, Datapoint, Dataset
from chroma_datasets.utils import transform_data, load_huggingface_dataset, to_chroma_schema
from langchain.text_splitter import CharacterTextSplitter
from typing import Optional, Union, Sequence, Dict, Mapping, List


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
