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

    @classmethod
    def load_data(cls):
        cls.hf_data = load_dataset("glue", "ax", split="test")

    @classmethod
    def raw_text(cls) -> str:
        if cls.hf_data is None:
            cls.load_data()
        return "\n".join(cls.hf_data["premise"])
    
    @classmethod
    def chunked(cls) -> List[Datapoint]:
        if cls.hf_data is None:
            cls.load_data()

        mapping = {
            "id": lambda row: str(row["idx"]),
            "metadata": lambda row: {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]},
            "document": "premise"
        }
        return transform_data(cls.hf_data, mapping)
    
    @classmethod
    def to_chroma(cls) -> AddEmbedding:
        return to_chroma_schema(cls.chunked())
