import os
import asyncio
import discord
from concurrent.futures import ThreadPoolExecutor


from modal import Image, Mount, Secret, Stub, asgi_app
from utils import pretty_log
from app import qanda

#Heavily inspired by https://github.com/teknium1/alpaca-discord/blob/main/alpaca-discord.py

image = Image.debian_slim(python_version="3.10").pip_install(  # and we install the following packages:
    "pynacl", 
    "requests",
    "discord==2.3.2",
    "langchain==0.0.314",  #0.0.184 
    # 🦜🔗: a framework for building apps with LLMs
    "openai~=0.28.1",      #0.27.7
    # high-quality language models and cheap embeddings
    "tiktoken==0.5.1",
    # tokenizer for OpenAI models
    "faiss-cpu==1.7.4",
    # vector storage and similarity search
    "pymongo[srv]==3.11",
    # python client for MongoDB, our data persistence solution
    "gradio~=3.34",
    # simple web UIs in Python, from 🤗
    "gantry==0.5.6",
    "requests"
    # 🏗️: monitoring, observability, and continual improvement for ML systems
)


secrets = [Secret.from_name("docchat-frontend-secret")]
intents = discord.Intents.all()
intents.members = True

client = discord.Client(intents=intents)
queue = asyncio.Queue()

stub = Stub(
    "docchat-discord",
    image=image,
    secrets=secrets,
    mounts=[Mount.from_local_python_packages("utils")],
)
@stub.function(
    # keep one instance warm to reduce latency, consuming ~0.2 GB while idle
    # this costs ~$3/month at current prices, so well within $10/month free tier credit
    keep_warm=1,
)

def start_client():
    key = os.environ.get("DISCORD_AUTH")
    pretty_log('Starting Discord client')

    client.run(key)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    asyncio.get_running_loop().create_task(background_task())    # This makes issues.

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if isinstance(message.channel, discord.channel.DMChannel) or (client.user and client.user.mentioned_in(message)):
        pretty_log(f'RECEIVED MESSAGE: {message.content} from {message.author.name}')
        await queue.put(message)

#@stub.function()
async def background_task():
    executor = ThreadPoolExecutor(max_workers=1)    
    loop = asyncio.get_running_loop()
    print("Task Started. Waiting for inputs.")
    while True:
        msg: discord.Message = await queue.get()
        username = client.user.name
        user_id = client.user.id
        message_content = msg.content.replace(f"@{username} ", "").replace(f"<@{user_id}> ", "")
        query = generate_prompt(message_content)
        pretty_log(f'!!!!!!!!!Message: {message_content}\n query: {query}')

        response = await send_request_to_backend(query)
        
        await msg.reply(response, mention_author=False)
        

def generate_backend_url():
    '''Generate the URL for the backend as you see it in the Modal UI.'''
    MODAL_USER_NAME = os.environ.get("MODAL_USER_NAME")
    MODAL_BACKEND_NAME = os.environ.get("MODAL_BACKEND_NAME")
    MODAL_ENDPOINT_NAME = os.environ.get("MODAL_ENDPOINT_NAME")
    MODAL_ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT")
    
    if None in (MODAL_USER_NAME, MODAL_BACKEND_NAME, MODAL_ENDPOINT_NAME, MODAL_ENVIRONMENT):
        pretty_log('Error generating backend URL.One or more required environment variables are not set.')
        raise ValueError("One or more required environment variables are not set.")
    
    url = f"https://{MODAL_USER_NAME}-{MODAL_ENVIRONMENT}--{MODAL_BACKEND_NAME}-{MODAL_ENDPOINT_NAME}.modal.run"
    
    pretty_log(f'Returning backend URL: {url}')
    return url


#@stub.function()
async def send_request_to_backend(query: str):
    import requests
    import json

    url = generate_backend_url()
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"query": query})

    # Log the request details
    pretty_log(f"""Request URL: {url}\nRequest Headers: {headers}\nRequest Data: {data}""")
    
    # Send a POST request to the backend
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        # If the status code is 200 (OK), you can parse the JSON response
        response_data = response.json()
        pretty_log(f"Request successful. Response: {response_data}")
        answer = response_data.get("answer")
        pretty_log(f"Answer: {answer}")
    else:
        pretty_log(f"Request failed. Response: {response}")
        answer = "I dunno"
    return answer



def generate_prompt(query):
    return f"""### Instruction:
You are an expert in sound and music computing and you use your enjoy sharing your knowledge with students currently following a masters in this topic to help them in their learning journey.
### Query:
{query}

### Response:
"""


def generate_prompt_okto(query):
    return f"""### Instruction:
You are an expert in electrical power transformers and using physiscs-inspired modelling to describe their operation.
### Query:
{query}

### Response:
"""