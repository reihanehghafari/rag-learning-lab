import requests
import numpy as np
from pathlib import Path



OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:1.5b"

DOCUMENT_PATH = Path("documents/doc.txt")




def load_document():
    return DOCUMENT_PATH.read_text(encoding="utf-8")




def create_chunks(text, chunk_size=3, overlap=1
                  ):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks




def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]




def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator




def retrieve(question, chunks, embeddings, top_k=3):

    question_embedding = get_embedding(question)

    scores = []

    for chunk, embedding in zip(chunks, embeddings):
        score = cosine_similarity(
            question_embedding,
            embedding
        )

        scores.append((score, chunk))

    scores.sort(reverse=True, key=lambda item: item[0])

    return scores[:top_k]




def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        chunk for score, chunk in retrieved_chunks
    )

    prompt = f"""
Answer the question using only the provided context.


say that you cannot find the answer in the documents answer in japanese

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]




def main():

    print("Loading document...")

    document = load_document()

    print("Creating chunks...")

    chunks = create_chunks(document)

    print("Generating embeddings...")

    embeddings = [
        get_embedding(chunk)
        for chunk in chunks
    ]
    print(embeddings)
    print(f"Created {len(chunks)} chunks.")

    question = input("\nAsk a question: ")

    retrieved = retrieve(
        question,
        chunks,
        embeddings
    )

    print("\nRetrieved chunks:")

    for score, chunk in retrieved:
        print(f"\nSimilarity: {score:.4f}")
        print(chunk)

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        retrieved
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
