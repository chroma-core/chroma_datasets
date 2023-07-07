__version__ = "0.1.5"

# Datasets
# from .datasets.state_of_the_union import StateOfTheUnion
from .dataset import Dataset


ef_instruction_dict = {
    "OpenAIEmbeddingFunction": """
        from chromadb.utils import embedding_functions
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key="YOUR_API_KEY",
            model_name="text-embedding-ada-002"
        )
    """
}


class StateOfTheUnion(Dataset):
    """
    This dataset contains the text of the 2022 State of the Union Address.

    Columns:
        - text: The text of the 2022 State of the Union address.
    """
    hf_data = None
    hf_dataset_name = "chromadb/state_of_the_union"
    embedding_function = "OpenAIEmbeddingFunction"
    embedding_function_instructions = ef_instruction_dict[embedding_function]


class HubermanPodcasts(Dataset):
    """
    5 Huberman podcasts on exercise. Source links are in the metadata.
    Contributors: Dexa AI (dexa.ai)

    1. https://dexa.ai/huberman/episodes/doc_1931
    2. https://dexa.ai/huberman/episodes/doc_2065
    3. https://dexa.ai/huberman/episodes/doc_2078
    4. https://dexa.ai/huberman/episodes/doc_5521
    5. https://dexa.ai/huberman/episodes/doc_2140
    """
    hf_data = None
    hf_dataset_name = "dexaai/huberman_on_exercise"
    embedding_function = "OpenAIEmbeddingFunction"
    embedding_function_instructions = ef_instruction_dict[embedding_function]


class PaulGrahamEssay(Dataset):
    """
        http://www.paulgraham.com/worked.html
    """
    hf_data = None
    hf_dataset_name = "chromadb/paul_graham_essay"
    embedding_function = "OpenAIEmbeddingFunction"
    embedding_function_instructions = ef_instruction_dict[embedding_function]