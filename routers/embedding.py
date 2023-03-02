# Clase en vídeo: https://youtu.be/_y9qQZXE24A?t=12475

### Products API ###
import pandas as pd
import numpy as np
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter(prefix="/embbeding_corpus",
                   tags=["embbeding"],
                   responses={404: {"message": "No encontrado"}})


# @router.get("/")
# async def products():
#     return products_list
class Item(BaseModel):
    text: str

class Question(BaseModel):
    pregunta: str
   

# @router.get("/{id}")
# async def products(id: int):
#     return products_list[id]

## Scrapping de pagina web
@router.post("/scrap_url")
async def scrap_url(url:str):
    print(url)
    return url


## procesamiento de incrustamiento
@router.post("/")
async def embedding(body: Item):
    #procesamiento de parrafos
    
    print(body.text)
    newtext = save_text_in_csv(body.text)
    
    parrafos = embedding_processing(newtext)
    #return parrafos
 
## procesamiento de archivos??
@router.post("/files")
async def scrap_url(url:str):
    print(url)
    return url

## embedding pregunta + completion (prueba de asistente)
@router.post("/completion")
async def completion(question:Question):
    df = pd.read_csv('embeddings2.csv', index_col=0)
    #print(df[0].values)
    answer_question(df, question="¿Que sabes de GamerGenius?")

    return question.pregunta

def remove_lines(text: str):
    
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    return text

def save_text_in_csv(text:str):
    # Create a list to store the text files
    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens = 500
    array_text=[text]
    # Create a dataframe from the list of texts
    df = pd.DataFrame(array_text, columns = [ 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = remove_lines(df.text)
    df.to_csv('scraped.csv')
    df.head() 
    
    shortened = []
    df = tokenization()
            # Loop through the dataframe
    for row in df.iterrows():

                # If the text is None, go to the next row
     if row[1]['text'] is None:
        continue

     print(row)           # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
                
                # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )
    
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()
    return df

        

def tokenization():
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv('scraped.csv', index_col=0)
    df.columns = [ 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    print(df['n_tokens'])
    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()
    return df
   

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Split the text into sentences
    sentences = text.split('.\n\n')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

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
                

           

def embedding_processing(parrafos):
    #api_url = "https://api.openai.com/v1/embeddings"
    openai_embedding_model = "text-embedding-ada-002"
    openai.api_key  = "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"
    parrafos['embeddings'] = parrafos.text.apply(lambda x: openai.Embedding.create(input=x, engine=openai_embedding_model)['data'][0]['embedding'])
    parrafos.to_csv('embeddings.csv')
    parrafos=pd.read_csv('embeddings.csv', index_col=0)
    parrafos['embeddings'] = parrafos['embeddings'].apply(eval).apply(np.array)
    parrafos.head()
    
    
    return parrafos

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Crea un contexto para una pregunta encontrando los parrafos mas similares desde el dataframe
    """

    # Get the embeddings for the question
    openai.api_key  = "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"
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    #print(q_embeddings)
    #print(df)
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


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
    max_tokens=150,
    stop_sequence=None
):
    """
    Responde la pregunta basado en el contexdo de los datos
    """
    
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Contexto:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        openai.api_key  = "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"
        response = openai.Completion.create(
            prompt=f"Responde la pregunta basado en el siguiente contexto, y si no sabes la respuesta dime  \"Eres un puto genio\"\n\nContexto: {context}\n\n---\n\nPreguntan: {question}\nRespuesta:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


