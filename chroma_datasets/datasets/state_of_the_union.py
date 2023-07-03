from chroma_datasets.types import AddEmbedding, Datapoint, Dataset
from chroma_datasets.utils import transform_data, load_huggingface_dataset, to_chroma_schema
from langchain.text_splitter import CharacterTextSplitter
from typing import Optional, Union, Sequence, Dict, Mapping, List


class StateOfTheUnion(Dataset):
    """
    This dataset contains the text of the 2022 State of the Union Address.

    Columns:
        - text: The text of the 2022 State of the Union address.
    """
    hf_data = None
    embedding_function = None

    @classmethod
    def load_data(cls):
        cls.hf_data = load_huggingface_dataset(
            "chromadb/state_of_the_union",
            split_name="data"
        )

    @classmethod
    def raw_text(cls) -> str:
        if cls.hf_data is None:
            cls.load_data()
        return "\n".join(cls.hf_data["text"])
    
    @classmethod
    def chunked(cls) -> List[Datapoint]:
        if cls.hf_data is None:
            cls.load_data()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        split = text_splitter.split_text(cls.raw_text())

        # convert to array of Datapoint types
        split = [
            {
                "id": str(i),
                "document": chunk
            }
            for i, chunk in enumerate(split)
        ]

        return split
    
    @classmethod
    def to_chroma(cls) -> AddEmbedding:
        return to_chroma_schema(cls.chunked())
