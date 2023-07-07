# imports
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, Dict, Mapping, List, Any
from typing_extensions import TypedDict
from chroma_datasets.types import AddEmbedding, Datapoint
from chroma_datasets.utils import load_huggingface_dataset, to_chroma_schema


class Dataset(ABC):
    """
        Abstract class for a dataset

        All datasets should inherit from this class

        Properties:
            hf_data: the raw data from huggingface
            embedding_function: the embedding function used to generate the embeddings
            embeddingFunctionInstructions: tell the user how to set up the embedding function
    """
    hf_dataset_name: str
    hf_data: Any
    embedding_function: str
    embedding_function_instructions: str

    @classmethod
    def load_data(cls):
        cls.hf_data = load_huggingface_dataset(
            cls.hf_dataset_name,
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