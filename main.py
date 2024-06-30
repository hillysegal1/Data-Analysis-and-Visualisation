!pip install transformers sentence-transformers datasets pinecone-client tqdm
!pip install cohere

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset, Dataset
import pinecone
import os
from tqdm import tqdm
import numpy as np
import warnings
from IPython.display import display
import cohere
import pandas as pd
warnings.filterwarnings("ignore")

# loading API keys from text files
with open("pinecone_api_key.txt") as f:
    PINECONE_API_KEY = f.read().strip()
with open("cohere_api_keys.txt") as f:
    COHERE_API_KEY = f.read().strip()

# choosing an embedding model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
model = SentenceTransformer(EMBEDDING_MODEL)

#LOADING AND PREPARING DATASET: 
def load_and_prepare_dataset(dataset_name: str, config_name: str, split: str, rec_num: int = 40) -> tuple:
    """
    Load a dataset and prepare a new column with concatenated descriptions from search results.
    Args:
        dataset_name: The name of the dataset to load
        config_name: The configuration name of the dataset
        split: The split of the dataset to load
        rec_num: The number of records to load
    Returns:
        tuple: A tuple containing the modified dataset and the list of descriptions
    """
    print("Loading and preparing the dataset")

    # loading the dataset
    dataset = load_dataset(dataset_name, config_name, split=split)

    # extracting the necessary number of records
    dataset = dataset.select(range(rec_num))

    # creating a new column 'description' by concatenating all descriptions from 'search_results'
    descriptions = []
    for item in dataset['search_results']:
        if 'description' in item:
            concatenated_description = " ".join(desc for desc in item['description'])
        else:
            concatenated_description = "No description found"
        descriptions.append(concatenated_description)

    # converting dataset to pandas DataFrame in order to add the new column
    df = pd.DataFrame(dataset)
    df['description'] = descriptions

    # converting back to Dataset
    dataset = Dataset.from_pandas(df)

    print("Dataset prepared with 'description' column")
    return dataset, descriptions

# loading and preparing the dataset
dataset_name = 'trivia_qa'
config_name = 'rc'
text_field = 'description'
rec_num = 40

dataset, descriptions = load_and_prepare_dataset(dataset_name, config_name, 'train', rec_num)

# EMBEDDING DATA:
def embed_dataset(
        descriptions: list,
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
) -> np.ndarray:
    """
    Embed the descriptions using a sentence-transformer model.
    Args:
        descriptions: The list of descriptions to embed
        model: The model to use for embedding
    Returns:
        np.ndarray: The embeddings
    """
    print("Embedding the descriptions")

    # embedding the descriptions
    embeddings = model.encode(descriptions, show_progress_bar=True)

    print("Done!")
    return embeddings

# embedding the dataset
embeddings = embed_dataset(descriptions, model=SentenceTransformer('all-MiniLM-L6-v2'))

# printing the shape of embeddings for verification
print(embeddings.shape)

# visualising dataset
pd_dataset = dataset.to_pandas()
pd_dataset.head(5)

# CREATING VECTOR DATABASE:
def create_pinecone_index(
        index_name: str,
        dimension: int,
        metric: str = 'cosine',
):
    """
    Create a pinecone index if it does not exist
    Args:
        index_name: The name of the index
        dimension: The dimension of the index
        metric: The metric to use for the index
    Returns:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """
    
    print("Creating a Pinecone index...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            # Remember! It is crucial that the metric you will use in your VectorDB will also be a metric your embedding
            # model works well with!
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    print("Done!")
    return pc

INDEX_NAME = 'trivia-qa-index'

# creating vector db
pc = create_pinecone_index(INDEX_NAME, embeddings.shape[1])

# ADDING DATA TO VECTOR DATABASE
def upsert_vectors(
        index: Pinecone,
        embeddings: np.ndarray,
        dataset: dict,
        text_field: str = 'description',
        batch_size: int = 128
):
    """
    Upsert vectors to a Pinecone index.

    Args:
        index: The Pinecone index object.
        embeddings: The embeddings to upsert.
        dataset: The dataset containing the metadata.
        text_field: The field in the dataset that contains the text for metadata.
        batch_size: The batch size to use for upserting.

    Returns:
        An updated Pinecone index.
    """
    print("Upserting the embeddings to the Pinecone index...")
    shape = embeddings.shape

    ids = [str(i) for i in range(shape[0])]
    meta = [{text_field: text} for text in dataset[text_field]]

    # creating list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeddings, meta))

    for i in tqdm(range(0, shape[0], batch_size)):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])

    print("Done upserting vectors!")
    return index

# upserting the embeddings to the Pinecone index
index = pc.Index(INDEX_NAME)
index_upserted = upsert_vectors(index, embeddings, dataset)

# LLM: 
def augment_prompt(
        query: str,
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
        index=None
) -> str:
    """
    Augment the prompt with the top 3 results from the knowledge base.

    Args:
        query: The query to augment.
        model: The model to use for encoding the query.
        index: The Pinecone index object.

    Returns:
        str: The augmented prompt.
    """
    # Encode the query using the sentence-transformer model
    query_vector = model.encode(query).tolist()

    # Get top 3 results from the Pinecone index
    query_results = index.query(
        vector=query_vector,
        top_k=3,
        include_values=False,
        include_metadata=True
    )['matches']

    # extracting the descriptions from the results, if available
    text_matches = [match['metadata'].get('description', '') for match in query_results]

    # filtering out empty descriptions
    text_matches = [desc for desc in text_matches if desc]

    # combining the text matches into a single string
    source_knowledge = "\n\n".join(text_matches)

    # creating the augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.
    Contexts:
    {source_knowledge}
    If the answer is not included in the source knowledge - say that you don't know.
    Query: {query}"""

    return augmented_prompt, source_knowledge

# example queries
queries = [
    "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    "In which decade did Billboard magazine first publish an American hit chart?",
    "Which city does David Soul come from?"
]

# initialising Cohere client
co = cohere.Client(api_key=COHERE_API_KEY)

# looping through each query and compare normal and augmented answers
for query in queries:
    print(f"NORMAL ANSWER for query: {query}")
    response_normal = co.chat(
        model='command-r-plus',
        message=query,
    )
    print(response_normal.text)
    
    print(f"AUGMENTED ANSWER for query: {query}")
    augmented_prompt, source_knowledge = augment_prompt(query, model=model, index=index)
    response_augmented = co.chat(
        model='command-r-plus',
        message=augmented_prompt,
    )
    print(response_augmented.text)
 
