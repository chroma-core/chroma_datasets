from chroma_datasets.types import AddEmbedding, Datapoint, Dataset
from chroma_datasets.utils import transform_data, load_huggingface_dataset, to_chroma_schema
from typing import Optional, Union, Sequence, Dict, Mapping, List
from datasets import load_dataset


class Glue(Dataset):
    """
    This dataset contains the popular Glue Dataset.
    It is hardcoded to "ax" subdataset and the "test" split.

    Columns:
        - premise: The premise of the sentence pair.
        - hypothesis: The hypothesis of the sentence pair.
        - label: The label of the sentence pair.
        - idx: The index of the sentence pair.
    """
    hf_data = None

    def __init__(self):
        self.hf_data = load_dataset("glue", "ax", split="test")

    def raw_text(self) -> str:
        return "\n".join(self.hf_data["premise"])
    
    def chunked(self) -> List[Datapoint]:
        mapping = {
            "id": lambda row: str(row["idx"]),
            "metadata": lambda row: {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]},
            "document": "premise"
        }
        return transform_data(self.hf_data, mapping)
    
    def to_chroma(self) -> AddEmbedding:
        return to_chroma_schema(self.chunked())
