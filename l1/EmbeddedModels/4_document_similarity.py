from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records."
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
# inside cosine both should be 2d

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])

print(query)
print(documents[index])
print("Similarity score is :", score)

# to store embeddings - - - vector dbs

