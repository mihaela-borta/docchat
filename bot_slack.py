from enum import Enum
from fastapi import Request, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import aiohttp
import json
import hashlib
import hmac

from modal import Image, Mount, Secret, Stub, asgi_app

from utils import pretty_log

image = Image.debian_slim(python_version="3.10").pip_install("pynacl", "requests", "slack_sdk")
slack_secrets = [Secret.from_name("docchat-frontend-slack-secret")]

#https://mihaela-borta-docchat--docchat-discord-bot.modal.run/interactions
#https://mihaela-borta-docchat--docchat-slack-bot_slack.modal.run/events
# https://mihaela-borta-dev--docchat-discord-bot-dev.modal.run
#f"https://{MODAL_USER_NAME}-{MODAL_ENVIRONMENT}--{MODAL_FRONTEND_NAME}-{MODULE_NAME}.modal.run/slack/events"



stub = Stub(
    "docchat-slack",
    image=image,
    secrets=slack_secrets,
    mounts=[Mount.from_local_python_packages("utils")],
)

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError


async def verify(request: Request):
    """Verify that the request is from Slack."""
    signing_secret = os.environ.get("SLACK_SIGNING_SECRET")

    body = await request.body()
    timestamp = request.headers.get('X-Slack-Request-Timestamp')    
    slack_signature = request.headers.get('X-Slack-Signature')

    # Create a basestring by concatenating timestamp and request body
    basestring = f'v0:{timestamp}:{body.decode("utf-8")}'

    # Calculate the HMAC-SHA256 hash using the signing secret
    signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        basestring.encode(),
        hashlib.sha256).hexdigest()
    
    # Compare the calculated signature with the incoming signature
    if not hmac.compare_digest(signature, slack_signature):
        pretty_log(f'Signature not matching:\nbasestring: {basestring}\nsignature: {signature}\nslack_signature: {slack_signature}')
        raise HTTPException(status_code=401, detail="Invalid request") from None
    
    return body


@stub.function(
    # keep one instance warm to reduce latency, consuming ~0.2 GB while idle
    # this costs ~$3/month at current prices, so well within $10/month free tier credit
    keep_warm=1,
)
@asgi_app(label="docchat-slack-bot")
def app() -> FastAPI:
    app = FastAPI()

    print('Hello from Slack!')

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.post("/slack/events") #For this you need to register an events URL in Slack. Te base is the URL which modal generates for your fronted
    async def handle_request(request: Request):
        "Verify incoming requests and if they're a valid command spawn a response."
        import urllib.parse
        body = await verify(request)
        data = body.decode('utf-8')
        data = urllib.parse.parse_qs(data)
        question = data["text"][0]
        user_name = data["user_name"][0]
        command = data["command"][0]
        channel_name = data["channel_name"][0]
        channel_id = data["channel_id"][0]
        #app_id = data["api_app_id"]
        token = os.environ.get('SLACK_BOT_TOKEN')

        pretty_log(f'channel_id: {channel_id} user_name: {user_name} question: {question}')

        if command == "/ask":
            respond.spawn(
                question,
                channel_id,
                user_name,
                token,
            )
            
        '''
            async with AsyncWebClient(token=token) as web_client:
                try:
                    # Craft your response message
                    response_text = "This is a response to the command /ask!"

                    # Send the response to the channel where the command originated
                    await web_client.chat_postMessage(
                        channel=channel_name,
                        text=response_text,
                    )
                except SlackApiError as e:
                    pretty_log(f"Error posting message: {e.response['error']}")

                    
        pretty_log(f"!!!! SLACK Request: {data}")
        
            
            respond.spawn(
                question,
                app_id,
                interaction_token,
                user_id,
            )
    
    
        '''
    return app
    


@stub.function()
async def respond(
    message: str,
    channel_id: str,
    user_name: str,
    bot_token: str,
): 
    
    try:
        response = message #await send_request_to_backend(question)
    except Exception as e:
        pretty_log("Error", e)
        response = construct_error_message(user_name)
    await send_message(response, channel_id, bot_token)


async def send_message(
    message: str,
    channel_id: str,
    bot_token: str,
):
    """Send a message to Slack."""
    url = f"https://slack.com/api/chat.postMessage"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {bot_token}'
    }
    payload = {
        'channel': channel_id,
        'text': message
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print("Message posted successfully:", result)
            else:
                print("Failed to post message:", response.status)



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


def construct_response(raw_response: str, user_id: str, question: str) -> str:
    """Wraps the backend's response in a nice message for Slack."""
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
    """Registers the slash command with Slack. Pass the force flag to re-register."""
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
                "type": SlackApplicationCommandOptionType.STRING.value,
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