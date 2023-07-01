from chroma_datasets.types import AddEmbedding, Datapoint, Dataset
from chroma_datasets.utils import transform_data, load_huggingface_dataset, to_chroma_schema
from langchain.text_splitter import CharacterTextSplitter
from typing import Optional, Union, Sequence, Dict, Mapping, List
import uuid


class SciQ(Dataset):
    """
    https://huggingface.co/datasets/sciq

    Columns:
        - question: str
        - correct_answer: str
        - support: str
        - distractor1: str
        - distractor2: str
        - distractor3: str
    """
    hf_data = None
    embedding_function = None

    def __init__(self):
        self.hf_data = load_huggingface_dataset(
            "sciq",
            split_name="test"
        )

    def raw_text(self) -> str:
        """
            Doesn't make sense for this dataset
        """
        raise NotImplementedError
    
    def chunked(self) -> List[Datapoint]:
        """
            The dataset is already chunked
        """
        dataset_rows = []
        for row in self.hf_data:

            # add questions
            dataset_rows.append({
                "id": str(uuid.uuid4()),
                "document": row["question"],
                "metadata": {  
                    "type": "question",
                }
            })

            # add supporting evidence
            dataset_rows.append({
                "id": str(uuid.uuid4()),
                "document": row["support"],
                "metadata": {
                    "type": "supporting_evidence"
                }
            })

        return dataset_rows
    
    def to_chroma(self) -> AddEmbedding:
        return to_chroma_schema(self.chunked())
