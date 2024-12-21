import os
from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, text_split
from src.helper import download_hugging_face_embeddings


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
print("Number of chunks:", len(text_chunks))
embeddings = download_hugging_face_embeddings()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
DIMENSION = int(os.getenv("DIMENSION"))

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


# Check if the index exists and create it only if it doesn't
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Index '{INDEX_NAME}' does not exist. Creating index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )
    print(f"Index '{INDEX_NAME}' created successfully.")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Connect to the existing or newly created index
index = pc.Index(INDEX_NAME)
print(f"Connected to index '{INDEX_NAME}'.")


# Upsert data into Pinecone
def upsert_embeddings_to_pinecone(index, text_chunks, embeddings, batch_size=100):
  # Extract text content from each chunk
  chunk_texts = [t.page_content for t in text_chunks]  # t.page_content is the raw document text
  chunk_embeddings = embeddings.embed_documents(chunk_texts)  # Generate embeddings for the text chunks

  for i in range(0, len(chunk_embeddings), batch_size):
    batch = chunk_embeddings[i:i+batch_size]  # Create a batch of embeddings
    # Store the text content in the metadata under the 'text' key
    metadata = [{"text": chunk_texts[i+j]} for j in range(len(batch))]
    # Upsert the vectors along with their metadata (including text)
    vectors = [(f"id-{i+j}", batch[j], metadata[j]) for j in range(len(batch))]
    index.upsert(vectors)  # Upsert the vectors into Pinecone

  print("Upsert complete.")

upsert_embeddings_to_pinecone(index, text_chunks, embeddings)