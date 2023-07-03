from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, Dict, Mapping, List, Any
from typing_extensions import TypedDict


Metadata = Mapping[str, Union[str, int, float]]
Vector = Union[Sequence[float], Sequence[int]]


class Datapoint(TypedDict):
    id: Optional[str]
    metadata: Optional[Metadata]
    document: Optional[str] 
    embedding: Optional[Vector]


class AddEmbedding(TypedDict): 
    embeddings: Optional[List[Any]] 
    metadatas: Optional[List[Dict[Any, Any]]] 
    documents: Optional[List[str]] 
    ids: List[str]


class Dataset(ABC):
    """
        Abstract class for a dataset

        All datasets should inherit from this class

        Properties:
            hf_data: the raw data from huggingface
            embedding_function: the embedding function used to generate the embeddings
            embeddingFunctionInstructions: tell the user how to set up the embedding function
    """
    hf_data: Any
    embedding_function: str
    embedding_function_instructions: str

    @classmethod
    @abstractmethod
    def raw_text(cls) -> str:
        """
            Returns the dataset as one long string.
            Useful if you want to chunk it yourself.
            Note: you will reset chunk-specific metadata, etc if it exists.
        """
        pass

    @classmethod
    @abstractmethod
    def chunked(cls) -> List[Datapoint]:
        """
            Returns the dataset with a default chunking scheme
        """
        pass

    @classmethod
    @abstractmethod
    def to_chroma(cls) -> AddEmbedding:
        """
            Returns the dataset in a format that can be used by Chroma
        """
        pass
