import pandas as pd


import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import string
import random

############ SERIE => sus elementos tienen que ser del mismo tipo.     [i:valor, i2:valor2, i3:valor3]
serie1 = pd.Series(['Matemáticas', 'Historia', 'Economía', 'Programación', 'Inglés'], dtype='string')
#print(f"############### SERIE 1 \n {serie1}")
#print(serie1.size) #numero de atributos de la serie
#print(serie1.index) #nombres de las filas de la serie
#print(serie1.dtype) #tipo de datos de los elementos de la serie.

s = pd.Series({'Matemáticas': 6.0,  'Economía': 4.5, 'Programación': 8.5})
#print(f"############### SERIE 2 \n {serie2}")
#print(serie2.size) #numero de atributos de la serie
#print(serie2.index) #nombres de las filas de la serie
#print(serie2.dtype) #tipo de datos de los elementos de la serie.


print(s[0:3], s.sum(), s.count())

print(s['Matemáticas'])

######### DATAFRAME => formato tabular. tiene, columnas y nombre de filas.
######### DataFrame(data=diccionario, index=filas, columns=columnas, dtype=tipos)

datos = {'nombre':['María', 'Luis', 'Carmen', 'Antonio'],
        'edad':[18, 22, 20, 21],
        'grado':['Economía', 'Medicina', 'Arquitectura', 'Economía'],
        'correo':['maria@gmail.com', 'luis@yahoo.es', 'carmen@gmail.com', 'antonio@gmail.com']
         }

df = pd.DataFrame(datos)
print(df)

tokenizer = tiktoken.get_encoding("cl100k_base")
df = pd.read_csv('https://raw.githubusercontent.com/asalber/manual-python/master/datos/colesterol.csv')
df['n_token_nombre'] = df.nombre.apply(lambda x: len(tokenizer.encode(x)))
#df['nueva_columna'] = [string.ascii_lowercase]*14
#df['n_tokennueva_columna'] = df.nueva_columna.apply(lambda x: len(tokenizer.encode(x)))
print(df[0:4])


df = pd.read_csv('scraped.csv', index_col=0)
print('##################### CSV DATAFRAME #####################')
print(df)
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
print(df)
print('Row count is:',df.shape[0])




#################


df = pd.read_csv('scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()
print(f"COLUMNAS DEL CSV {df.columns}")

################################################################################
### Step 8
################################################################################

max_tokens = 500


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')
    # print(f"##################### SENTENCES ================ > \n {sentences}")
    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    print(f"N_TOKENS =========== > \n {n_tokens}")
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        print(f"SENTENCE => {sentence}, TOKEN => {token}")
        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


shortened = []

# Loop through the dataframe
for row in df.iterrows():
    print(f"ROW ===============> \n {row}")
    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])

################################################################################
### Step 9
################################################################################

df = pd.DataFrame(shortened, columns=['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
# print(f"##################### DF[N_TOKENS] ================ > \n {df['n_tokens']}")
df.n_tokens.hist()

################################################################################
### Step 10
################################################################################

# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
openai.api_key  = "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"
df['embeddings'] = df.text.apply(
    lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('embeddings.csv')
df.head()

################################################################################
### Step 11
################################################################################

df = pd.read_csv('embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()


################################################################################
### Step 12
################################################################################

def create_context(
        question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    openai.api_key  = "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"
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # print(f"SUMA FILAS EMBEDING {df.shape[0]}")
    print(f"SUMA FILAS EMBEDING {len(df.columns)}")
    print(f"DF['EMBEDINGS]' ================== > \n  COUNT TOTAL ROW EMBEDING {df['embeddings']} - {df['embeddings']}")
    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    print(f"DF['EMBEDINGS].VALUES' ================== > \n {df['embeddings'].values}")

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=1000,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        openai.api_key  = "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"
        response = openai.Completion.create(
            # prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            prompt=f"Explayate respondiendo la pregunta segun el siguiente contexto, si no se puede,  \"no lo se\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0.8,
            max_tokens=max_tokens,
            # N° maximo de tokens, pueden ser hasta 2048 entre la inicializacion y la finalizacion. (promt + respuesta = 4000)
            top_p=1,
            # controla diversidad a travez del muestreo del nucleo. 0.5 considera la mitad de las opcioenes ponderadas por probabilidad. [0,1]
            frequency_penalty=1.5,
            # cuanto penalizar nuevos tokens, en funcion de su frecuencia existente hasta el momento. disminuye la probabilidad de repetir del texto palarba por palabra. [0,2]
            presence_penalty=2,
            # Cuanto penalizar nuevos tokens en funcion de si aparecen en el texto hasta el momento. Aumenta la probabilidad de que el modelo hable sobre nuevos temas [0,2]
            stop=stop_sequence,  # te dice de donde parar por ejemplo, donde dividir-
            model=model,
            # best_of=5 # [1,20]
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


################################################################################
### Step 13
################################################################################

print(answer_question(df,
                      question="respondeme las siguientes preguntas: \n 1. Servicios que presta la empresa. \n 2. que soluciones se ofrecen. \n 3. hazme un resumen del contexto",
                      debug=False))
