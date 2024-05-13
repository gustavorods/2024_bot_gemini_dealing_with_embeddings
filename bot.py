# Importing necessary libraries
import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

# Loading environment variables
load_dotenv()

# Configuring API key
genai.configure(api_key=os.getenv('API_KEY'))

# Setting up model generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

# Safety settings for content generation
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initializing the generative model
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Example documents
document1 = {
    "Title": "comer arroz faz bem pra saúde",
    "Content": "Segundo especialistas, comer arroz faz muito bem pra saúde!"
}

document2 = {
    "Title": "Benefícios do exercício físico para a saúde",
    "Content": "De acordo com estudos recentes, a prática regular de exercícios físicos traz inúmeros benefícios para a saúde."
}

document3 = {
    "Title": "Dormir bem melhora o desempenho cognitivo",
    "Content": "Pesquisadores afirmam que uma boa noite de sono está diretamente relacionada ao melhor desempenho cognitivo."
}

# List of documents
documents = [document1, document2, document3]

# Creating a DataFrame from documents
df = pd.DataFrame(documents)
df.columns = ["Title", "Content about"]

# Output: Display DataFrame
print(df)

# Embedding model path
embedding_model = "models/embedding-001"

# Function to embed content
def embed_content(title, text):
    return genai.embed_content(model=embedding_model,
                               content=text,
                               title=title,
                               task_type="RETRIEVAL_DOCUMENT")["embedding"]

# Output: Add embeddings to DataFrame
df["Embeddings"] = df.apply(lambda row: embed_content(row["Title"], row["Content about"]), axis=1)
print(df)

# Function to generate and search query
def generate_and_search_query(query, base, model):
    query_embedding = genai.embed_content(model=model,
                                          content=query,
                                          task_type="RETRIEVAL_QUERY")["embedding"]
    dot_products = np.dot(np.stack(df["Embeddings"]), query_embedding)
    index = np.argmax(dot_products)
    return df.iloc[index]["Content about"]

# Output: Generate and search query
query_result = generate_and_search_query("Como se alimentar melhor", df, embedding_model)
print(query_result)

# Prompt for rewriting text
prompt2 = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {query_result}"

# Initializing generative model for rewriting
model_2 = genai.GenerativeModel("gemini-1.0-pro", generation_config=generation_config)

# Output: Rewrite text
response = model_2.generate_content(prompt2)
print(response.text)
