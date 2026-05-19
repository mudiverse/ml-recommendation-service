#fastAPi Service
#1. load trained files
#2. expose recommendation API
#3. return Movie IDs
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
import faiss
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#load the saved files
movies = pd.read_pickle("movies.pkl")

embeddings = pickle.load(open("embeddings.pkl", "rb"))

index = faiss.read_index("faiss_index.bin")
print("Model loaded :")

#API : ENDpoits

@app.get("/")
def home():
    return{
        "message":"Recommendation System"
    }


#recoomend endpoint

@app.get("/recommend/{movie_id}")
def recommend(movie_id: int):   #handler function
    movie_row = movies[
        movies["id"]==movie_id
    ]

    if(movie_row.empty):
        return{
            "error":"Movie not found"
        }
    
    idx = movie_row.index[0]

    #query embedding
    query_vec = np.array(
        [embeddings[idx]]
    ).astype("float32")

    #similarity search
    distance , indices = index.search(query_vec, k=4) #find nearby 4 movies

    recommended =[]

    for i in indices[0][1:]:
        recommended.append(int(movies.iloc[i]["id"]))
    
    return recommended


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

