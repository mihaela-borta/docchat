from enum import Enum
from fastapi import Request, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import aiohttp
import json
import requests
import modal
from typing import List, Dict, Optional, Tuple

from modal import Image, Mount, Secret, Stub, asgi_app

from utils import pretty_log

image = Image.debian_slim(python_version="3.10").pip_install("pynacl", "requests")
discord_secrets = [Secret.from_name("docchat-frontend-secret")]

stub = Stub(
    "docchat-discord",
    image=image,
    secrets=discord_secrets,
    mounts=[Mount.from_local_python_packages("utils")],
)


class DiscordInteractionType(Enum):
    PING = 1  # hello from Discord
    APPLICATION_COMMAND = 2  # an actual command

class DiscordResponseType(Enum):
    PONG = 1  # hello back
    DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE = 5  # we'll send a message later

class DiscordApplicationCommandOptionType(Enum):
    STRING = 3  # with language models, strings are all you need


@stub.function(
    # keep one instance warm to reduce latency, consuming ~0.2 GB while idle
    # this costs ~$3/month at current prices, so well within $10/month free tier credit
    keep_warm=1,
)
@asgi_app(label="docchat-discord-bot")
def app() -> FastAPI:
    app = FastAPI()

    print('Hello from Discord!')

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.post("/interactions") #For this you need to register an interactions URL in Discord. Te base is the URL which modal generates for your fronted
    async def handle_request(request: Request):
        "Verify incoming requests and if they're a valid command spawn a response."
        pretty_log(f'!!!! Received request: {request}')

        # while loading the body, check that it's a valid request from Discord
        body = await verify(request)
        data = json.loads(body.decode())

        if data.get("type") == DiscordInteractionType.PING.value:
            return {"type": DiscordResponseType.PONG.value}

        if data.get("type") == DiscordInteractionType.APPLICATION_COMMAND.value:
            # this is a command interaction
            app_id = data["application_id"]
            interaction_token = data["token"]
            user_id = data["member"]["user"]["id"]
            user_name = data["member"]["user"]["username"]

            query = data["data"]["options"][0]["value"]
            message_id = data['id']
            channel_id = data['channel_id']

            pretty_log(f'Processing query: {query}')

            # kick off our actual response in the background
            respond.spawn(
                query,
                interaction_token,
                app_id,
                user_id,
                user_name,
                message_id,
                channel_id,
            )
            # and respond immediately to let Discord know we're on the case
            return {
                "type": DiscordResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE.value
            }

        raise HTTPException(status_code=400, detail="Bad request")

    return app


@stub.function()
async def respond(
    query: str,
    interaction_token: str,
    application_id: str,
    user_id: str,
    user_name: str,
    message_id: str,
    channel_id: str,
): 
    try:
        raw_response = await send_query_request(message_id, query)
        await log_query(channel_id, user_name, message_id)
        response = construct_response(raw_response, user_id, query)
        response_id = await send_response(response, application_id, interaction_token)
        pretty_log(f'Sent response: {response}\n Message ID: {response_id}')
        await log_query_answer(channel_id, message_id, response_id)

    except Exception as e:
        pretty_log("Error", str(e))
        response = construct_error_message(user_id)


async def send_response(
    response: str,
    application_id: str,
    interaction_token: str,
):
    # TODO: separate function for creating the payload. Make the interaction_url a global variable
    # TODO: make data classes for messages?

    interaction_url = (f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}")
    payload = format_http_payload(response)

    async with aiohttp.ClientSession() as session:
        async with session.post(interaction_url, data=payload) as resp:
            response = await resp.text()
            response = json.loads(response)
            return response['id']


async def send_query_request(id: str, query: str) -> str:
    data = json.dumps({"request_type": "query", "message_id": id, "query": query})
    response = send_request_to_backend(data=data)

    if response.status_code == 200:
        answer = response.json()
        pretty_log(f"Request successful. Response: {answer}")
        pretty_log(f"Answer: {answer}")
    else:
        pretty_log(f"Request failed. Response: {response}")
        answer = "I dunno"
    return answer


def send_request_to_backend(data: str) -> str:
    url = generate_backend_url()
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=data)
    return response


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
    
    return url


async def log_query(channel_id, user, message_id):
    data = json.dumps({"request_type": "log_query", "channel_id": channel_id, "user": user, "message_id": message_id})
    response = send_request_to_backend(data=data)
    pretty_log(f"Logged Discord message id for query in DB. Response: {response}")
    return response


async def log_query_answer(channel_id, query_id, answer_id):
    pretty_log(f"Logging Discord message ids for query and answer in DB.")
    data = json.dumps({"request_type": "log_query_answer",
                       "channel_id": channel_id,
                       "query_id": query_id,
                       'answer_id': answer_id})
    response = send_request_to_backend(data=data)
    pretty_log(f"Logged Discord message id for answer in DB. Response: {response}")
    return response


async def log_reaction(data: dict):
    data["request_type"] = "log_reaction" 
    data = json.dumps(data)
    response = send_request_to_backend(data=data)
    pretty_log(f"Logged reaction in DB. Response: {response}")
    return response


async def log_reactions_in_backend(messages: List[Dict]):
    if messages is not None and len(messages)>0:
        data = dict()
        for message in messages:
            if 'reactions' in message.keys():
                data['channel_id'] = message['channel_id'] 
                data['message_id'] = message['id']
                data['reactions'] = []
                for reaction in message['reactions']:
                    if 'emoji' in reaction.keys():
                        emoji = reaction['emoji']['name']
                        count = reaction['count']
                        data['reactions'].append({'emoji': emoji, 'count': count})
                response = await log_reaction(data)
                

async def get_guilds(discord_headers):
    # Get list of guilds the bot is part of
    guilds_url = "https://discord.com/api/v10/users/@me/guilds"
    response = requests.get(guilds_url, headers=discord_headers) #OK to use the request library here as this is a light weight call
    guilds = [guild['id']  for guild in response.json()]
    pretty_log(f"Guilds: {guilds}")
    return guilds 


async def get_channels(guilds, discord_headers):
    from itertools import chain
    channels = []
    
    for guild_id in guilds:
        channels_url = f"https://discord.com/api/v10/guilds/{guild_id}/channels" #OK to use the request library here as this is a light weight call
        response = requests.get(channels_url, headers=discord_headers)
        guild_channels = [channel['id'] for channel in response.json()]
        channels.append(guild_channels)

    channels = list(chain.from_iterable(channels))
    pretty_log(f"Channels: {channels}")
    return channels

@stub.function(schedule=modal.Period(minutes=60))
async def get_past_messages():
    import time
    BOT_TOKEN = os.environ.get("DISCORD_AUTH")

    # Get list of guilds and channels the bot is part of
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bot {BOT_TOKEN}"
    }
    guilds = await get_guilds(headers)
    time.sleep(2)
    channels = await get_channels(guilds, headers)

    for channel_id in channels:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response_text = await response.text()
                response_text = json.loads(response_text)
                if response.status == 200:
                    messages = await response.json()
                    await log_reactions_in_backend(messages)
                else:
                    pretty_log(f"!!!! Error fetching messages: {response_text}")
                    raise HTTPException(status_code=response.status, detail="Failed to fetch messages")


def format_http_payload(payload: str) -> str:
    json_payload = {"content": f"{payload}"}

    payload = aiohttp.FormData()
    payload.add_field(
        "payload_json", json.dumps(json_payload), content_type="application/json"
    )
    return payload


async def verify(request: Request):
    """Verify that the request is from Discord."""

    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError

    public_key = os.getenv("DISCORD_PUBLIC_KEY")
    verify_key = VerifyKey(bytes.fromhex(public_key))

    signature = request.headers.get("X-Signature-Ed25519")
    timestamp = request.headers.get("X-Signature-Timestamp")
    body = await request.body()

    for header, value in request.headers.items():
        pretty_log(f"Header: {header}, Value: {value}")

    message = timestamp.encode() + body
    try:
        verify_key.verify(message, bytes.fromhex(signature))
    except BadSignatureError:
        # IMPORTANT: if you let bad signatures through,
        # Discord will refuse to talk to you
        raise HTTPException(status_code=401, detail="Invalid request") from None

    return body


def construct_response(raw_response: str, user: str, query: str) -> str:
    """Wraps the backend's response in a nice message for Discord."""
    rating_emojis = {
        "👍": "if the response was helpful",
        "👎": "if the response was not helpful",
    }

    emoji_reaction_text = " or ".join(
        f"react with {emoji} {reason}" for emoji, reason in rating_emojis.items()
    )
    emoji_reaction_text = emoji_reaction_text.capitalize() + "."

    response = f"""<@{user}> asked: _{query}_

    Here's my best guess at an answer, with sources so you can follow up:

    {raw_response}

    Emoji react to let us know how we're doing!
    {emoji_reaction_text}
    """

    return response


def construct_error_message(user: str) -> str:
    error_message = (
        f"*Sorry <@{user}>, an error occured while answering your query."
    )

    if os.getenv("DISCORD_MAINTAINER_ID"):
        error_message += f" I've let <@{os.getenv('DISCORD_MAINTAINER_ID')}> know."
    else:
        pretty_log("No maintainer ID set")
        error_message += " Please try again later."

    error_message += "*"
    return error_message


@stub.function()
def create_slash_command(force: bool = True):
    """Registers the slash command with Discord. Pass the force flag to re-register."""
    BOT_TOKEN = os.getenv("DISCORD_AUTH")
    CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bot {BOT_TOKEN}",
    }
    url = f"https://discord.com/api/v10/applications/{CLIENT_ID}/commands"

    pretty_log(f"Creating slash command at {url}")

    command_description = {
        "name": "ask",
        "description": "Ask a query about music and sound computing",
        "options": [
            {
                "name": "query",
                "description": "A query about topics relevant for someone studying a masters in sound and music computing",
                "type": DiscordApplicationCommandOptionType.STRING.value,
                "required": True,
                "max_length": 200,
            }
        ],
    }

    pretty_log(command_description)
    # first, check if the command already exists
    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
    except Exception as e:
        raise Exception("Failed to create slash command") from e

    commands = response.json()

    command_exists = any(command.get("name") == "ask" for command in commands)

    # and only recreate it if the force flag is set
    if command_exists and not force:
        return

    response = requests.post(url, headers=headers, json=command_description)
    try:
        response.raise_for_status()
    except Exception as e:
        raise Exception("Failed to create slash command") from e