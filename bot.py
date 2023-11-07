from enum import Enum
from fastapi import Request, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import aiohttp
import json

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

        # while loading the body, check that it's a valid request from Discord
        body = await verify(request)
        data = json.loads(body.decode())

        pretty_log(f'!!!! VERIFYIED REQUEST: {data}')

        if data.get("type") == DiscordInteractionType.PING.value:
            # "ack"nowledge the ping from Discord
            return {"type": DiscordResponseType.PONG.value}

        if data.get("type") == DiscordInteractionType.APPLICATION_COMMAND.value:
            # this is a command interaction
            app_id = data["application_id"]
            interaction_token = data["token"]
            user_id = data["member"]["user"]["id"]

            question = data["data"]["options"][0]["value"]
            pretty_log(f'!!!!Processing command: {question}')

            # kick off our actual response in the background
            respond.spawn(
                question,
                app_id,
                interaction_token,
                user_id,
            )

            # and respond immediately to let Discord know we're on the case
            return {
                "type": DiscordResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE.value
            }

        raise HTTPException(status_code=400, detail="Bad request")

    return app



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


@stub.function()
async def respond(
    question: str,
    application_id: str,
    interaction_token: str,
    user_id: str,
): 
    
    try:
        response = await send_request_to_backend(question)
    except Exception as e:
        pretty_log("Error", e)
        response = construct_error_message(user_id)
    await send_response(response, application_id, interaction_token)    


async def send_response(
    response: str,
    application_id: str,
    interaction_token: str,
):
    """Send a response to the user interaction."""

    interaction_url = (
        f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}"
    )

    json_payload = {"content": f"{response}"}

    payload = aiohttp.FormData()
    payload.add_field(
        "payload_json", json.dumps(json_payload), content_type="application/json"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(interaction_url, data=payload) as resp:
            await resp.text()


async def verify(request: Request):
    """Verify that the request is from Discord."""

    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError

    public_key = os.getenv("DISCORD_PUBLIC_KEY")
    verify_key = VerifyKey(bytes.fromhex(public_key))

    signature = request.headers.get("X-Signature-Ed25519")
    timestamp = request.headers.get("X-Signature-Timestamp")
    body = await request.body()

    message = timestamp.encode() + body
    try:
        verify_key.verify(message, bytes.fromhex(signature))
    except BadSignatureError:
        # IMPORTANT: if you let bad signatures through,
        # Discord will refuse to talk to you
        raise HTTPException(status_code=401, detail="Invalid request") from None

    return body


def construct_response(raw_response: str, user_id: str, question: str) -> str:
    """Wraps the backend's response in a nice message for Discord."""
    rating_emojis = {
        "👍": "if the response was helpful",
        "👎": "if the response was not helpful",
    }

    emoji_reaction_text = " or ".join(
        f"react with {emoji} {reason}" for emoji, reason in rating_emojis.items()
    )
    emoji_reaction_text = emoji_reaction_text.capitalize() + "."

    response = f"""<@{user_id}> asked: _{question}_

    Here's my best guess at an answer, with sources so you can follow up:

    {raw_response}

    Emoji react to let us know how we're doing!

    {emoji_reaction_text}
    """

    return response


def construct_error_message(user_id: str) -> str:
    import os

    error_message = (
        f"*Sorry <@{user_id}>, an error occured while answering your question."
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
    import os
    import requests

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
        "description": "Ask a question about music and sound computing",
        "options": [
            {
                "name": "question",
                "description": "A question about topics relevant for someone studying a masters in sound and music computing",
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