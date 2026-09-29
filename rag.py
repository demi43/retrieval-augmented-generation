import re
# chunking 
# open the file and read its content
text=open("lagosgirlresturant.md",encoding="utf-8").read()

# cut the text where ever the line starts with ## or ### 
parts=re.split(r"\n(?=#{2,3})",text)

# 3. Keep only the pieces that start with a heading, and remove extra spaces
chunks=[p.strip() for p in parts if p.strip().startswith("#")]
#chunking is the sepreation based on the heading of the documents in the text 
print(len(chunks))
for chunk in chunks:
    print("-", chunk.splitlines()[0])

# embedding into vectors 

from sentence_transformers import SentenceTransformer

#loading an embedding model
embedder=SentenceTransformer("BAAI/bge-small-en-v1.5")

#turning chunks into a vector representation
chunk_vectors=embedder.encode(chunks,normalize_embeddings=True)

# print("shape:",chunk_vectors.shape)
# print("First 5 number of chunk 1:",chunk_vectors[0][:19])

import numpy as np

# retrieval

def retrieval(question,k=5):

    question_vector=embedder.encode([question],normalize_embeddings=True)[0]
    #calculating the cosine similarity between the question and the chunks
    scores=chunk_vectors @ question_vector
    #cosine similarities are sorted in descending order and the top k chunks are selected
    best=np.argsort(scores)[::-1][:k]
    
    return [(chunks[i],scores[i]) for i in best]
results =retrieval("Are you open on mondays?")
for chunk, score in results:
    print(round(float(score),2),chunk.splitlines()[0])

import ollama
# result using ai 
def anwser(question):
    found=retrieval(question)
    context="\n\n---\n\n".join(chunk for chunk, score in found)
    prompt = f"""You are the customer assistant for Lagos Girl Restaurant.
Speak AS the restaurant, using "we" and "our" (for example: "We are closed on Mondays").

Rules:
- Answer using ONLY the information below.
- Give a direct answer first (Yes/No if possible), then short details.
- Include ALL relevant details, such as extra fees or conditions.
- Never contradict yourself.
- If the answer is not in the information, say "I'm not sure about that. Please call us at (555) 012-3456."

<information>
{context}
</information>

Customer question: {question}"""


    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},   # more consistent, factual answers
    )
    return response["message"]["content"]
while True:
    q = input("\nAsk a question (or 'quit'): ")
    if q == "quit":
        break
    print(anwser(q))