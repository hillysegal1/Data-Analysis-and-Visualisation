
! pip install transformers sentence-transformers datasets cohere pinecone-client tqdm

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
import os
from tqdm import tqdm
import cohere
import numpy as np
import warnings
from IPython.display import display

warnings.filterwarnings("ignore")

# loading API keys from text files
with open("cohere_api_keys.txt") as f:
  COHERE_API_KEY = f.read().strip()
with open("pinecone_api_key.txt") as f:
  PINECONE_API_KEY = f.read().strip()

# choosing an embedding model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
model = SentenceTransformer(EMBEDDING_MODEL)

def load_and_embedd_dataset(
        dataset_name: str = 'JacquesVlaming/Questions_Answers',
        split: str = 'train',
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
        text_field: str = 'context',
        rec_num: int = 400
) -> tuple:
    """
    Load a dataset and embedd the text field using a sentence-transformer model
    Args:
        dataset_name: The name of the dataset to load
        split: The split of the dataset to load
        model: The model to use for embedding
        text_field: The field in the dataset that contains the text
        rec_num: The number of records to load and embedd
    Returns:
        tuple: A tuple containing the dataset and the embeddings
    """

    print("Loading and embedding the dataset")

    # Load the dataset
    dataset = load_dataset(dataset_name, 'default', split=split)

    # Embed the first `rec_num` rows of the dataset
    embeddings = model.encode(dataset[text_field][:rec_num])

    print("Done!")
    return dataset, embeddings

DATASET_NAME = 'JacquesVlaming/Questions_Answers'

# loading and embedding the dataset
dataset, embeddings = load_and_embedd_dataset(
    dataset_name=DATASET_NAME,
    rec_num=400,
    model=model,
)
shape = embeddings.shape

# visualising data
pd_dataset = dataset.to_pandas()
pd_dataset.head(5)

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
    from pinecone import Pinecone, ServerlessSpec
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

INDEX_NAME = 'questions-answers'

# creating vector database
pc = create_pinecone_index(INDEX_NAME, shape[1])

def upsert_vectors(
        index: Pinecone,
        embeddings: np.ndarray,
        dataset: dict,
        text_field: str = 'context',
        batch_size: int = 128
):
    """
    Upsert vectors to a pinecone index
    Args:
        index: The pinecone index object
        embeddings: The embeddings to upsert
        dataset: The dataset containing the metadata
        batch_size: The batch size to use for upserting
    Returns:
        An updated pinecone index
    """
    print("Upserting the embeddings to the Pinecone index...")
    shape = embeddings.shape

    ids = [str(i) for i in range(shape[0])]
    meta = [{text_field: text} for text in dataset[text_field]]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeddings, meta))

    for i in tqdm(range(0, shape[0], batch_size)):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])
    return index

# upserting the embeddings to the Pinecone index
index = pc.Index(INDEX_NAME)
index_upserted = upsert_vectors(index, embeddings, dataset)

# LLM:
def augment_prompt(
        query: str,
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
        index=None,
) -> str:
    """
    Augment the prompt with the top 2 results from the knowledge base
    Args:
        query: The query to augment
        index: The vectorstore object
    Returns:
        str: The augmented prompt
    """
    results = [float(val) for val in list(model.encode(query))]

    # retrieving top 2 results from knowledge base
    query_results = index.query(
        vector=results,
        top_k=2,
        include_values=True,
        include_metadata=True
    )['matches']
    text_matches = [match['metadata']['context'] for match in query_results]

    # get the text from the results
    source_knowledge = "\n\n".join(text_matches)

    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.
    Contexts:
    {source_knowledge}
    If the answer is not included in the source knowledge - say that you don't know.
    Query: {query}"""
    return augmented_prompt, source_knowledge

# queries to test
queries = ["What is the name of the book for Jonny Brock and Clare Gorst?","Who was a descendant of Genghis Khan?",
           ""What was the Electronic Thumb?"]

# initialising Cohere client
co = cohere.Client(api_key=COHERE_API_KEY)

# looping through each query and comparing normal and augmented answers
for query in queries:
    print("QUERY:",query)
    print(f"REGULAR ANSWER for query:")
    response_normal = co.chat(
        model='command-r-plus',
        message=query,
    )
    print(response_normal.text)

    print(f"AUGMENTED ANSWER for query:")
    augmented_prompt, source_knowledge = augment_prompt(query, model=model, index=index)
    response_augmented = co.chat(
        model='command-r-plus',
        message=augmented_prompt,
    )
    print(response_augmented.text)
    print("source knowledge",source_knowledge)
