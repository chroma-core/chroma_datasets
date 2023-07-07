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
