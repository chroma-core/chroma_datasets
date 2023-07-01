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

    def __init__(self):
        self.hf_data = load_huggingface_dataset(
            "chromadb/state_of_the_union",
            split_name="data"
        )

    def raw_text(self) -> str:
        return "\n".join(self.hf_data["text"])
    
    def chunked(self) -> List[Datapoint]:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        split = text_splitter.split_text(self.raw_text())

        # convert to array of Datapoint types
        split = [
            {
                "id": str(i),
                "document": chunk
            }
            for i, chunk in enumerate(split)
        ]

        return split
    
    def to_chroma(self) -> AddEmbedding:
        return to_chroma_schema(self.chunked())
