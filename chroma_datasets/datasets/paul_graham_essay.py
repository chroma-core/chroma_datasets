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
