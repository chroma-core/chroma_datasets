from chroma_datasets.types import AddEmbedding, Datapoint, Dataset
from chroma_datasets.utils import transform_data, load_huggingface_dataset, to_chroma_schema
from langchain.text_splitter import CharacterTextSplitter
from typing import Optional, Union, Sequence, Dict, Mapping, List


class HubermanPodcasts(Dataset):
    """
    5 Huberman podcasts on exercise. Source links are in the metadata.
    Contributors: Dexa AI (dexa.ai)

    1. https://dexa.ai/huberman/episodes/doc_1931
    2. https://dexa.ai/huberman/episodes/doc_2065
    3. https://dexa.ai/huberman/episodes/doc_2078
    4. https://dexa.ai/huberman/episodes/doc_5521
    5. https://dexa.ai/huberman/episodes/doc_2140

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
            "dexaai/huberman_on_exercise",
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
