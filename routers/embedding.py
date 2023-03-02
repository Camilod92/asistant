# Clase en vídeo: https://youtu.be/_y9qQZXE24A?t=12475

### Products API ###
import openai


from fastapi import APIRouter

router = APIRouter(prefix="/embbeding_corpus",
                   tags=["embbeding"],
                   responses={404: {"message": "No encontrado"}})


# @router.get("/")
# async def products():
#     return products_list


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
async def embedding(text: str):
    #procesamiento de parrafos
   
    texot = """¡Por supuesto! Aquí te presento una empresa ficticia del mundo de los videojuegos con su misión, visión, análisis FODA y descripción de la empresa y sus servicios.

Nombre de la empresa: GamerGenius

Misión: Proporcionar la mejor experiencia de juego posible a nuestros clientes, ofreciendo los videojuegos más populares y emocionantes, junto con una amplia gama de servicios complementarios para satisfacer todas las necesidades de los jugadores.

Visión: Ser líderes en el mercado de los videojuegos, reconocidos por nuestra pasión por los juegos, nuestro compromiso con la calidad y la innovación, y nuestra capacidad para satisfacer todas las necesidades de los jugadores."""
    #texot = texot.replace('""','')
    newtext = get_add_paras(texot,1)
    parrafos = embedding2(newtext)
    return parrafos
 
## procesamiento de archivos??
@router.post("/files")
async def scrap_url(url:str):
    print(url)
    return url

## embedding pregunta + completion (prueba de asistente)
@router.post("/completion")
async def completion(obj:object):
    print(obj)
    return obj

def get_add_paras(text: str, reqblank: str):
    parastoadd = []
    
    min_para_words = 5
    if reqblank == 1:
        rawparas = text.split('\n\n')
    else:
        rawparas = text.split('\n')
    
    for rawpara in rawparas:
        
        rawpara = rawpara.strip().replace('\n', ' ')
     
        #if rawpara[-1] != '?' and len(rawpara.split()) >= min_para_words:
        parastoadd.append(rawpara)

    return parastoadd

def embedding2(parrafos):
    api_url = "https://api.openai.com/v1/embeddings"
    openai_embedding_model = "text-embedding-ada-002"
    openai.api_key  = "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"
    print(parrafos)
    headers = {
            "Authorization": f"Bearer ",
            "Content-Type": "application/json",
        }
    payload = {
            "model": openai_embedding_model,
            "input": parrafos,
        }
    response = openai.Embedding.create(input=parrafos, engine='text-embedding-ada-002')['data'][0]['embedding']

    
    print(f"Response:{response}")
    #return response
    
    
    
    # with httpx.AsyncClient() as client:
    #     headers = {
    #         "Authorization": f"Bearer {api_key}",
    #         "Content-Type": "application/json",
    #     }
    #     payload = {
    #         "model": openai_embedding_model,
    #         "input": parrafos,
    #     }
    #     response =  client.post(api_url, headers=headers, json=payload)
    #     print(f"Response:{response}")
    #     return response
    
    