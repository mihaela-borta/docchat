"""Builds a CLI, Webhook, and Gradio app for Q&A on the Full Stack corpus.

For details on corpus construction, see the accompanying notebook."""
import modal

import vecstore
from utils import pretty_log


# definition of our container image for jobs on Modal
# Modal gets really powerful when you start using multiple images!
image = modal.Image.debian_slim(  # we start from a lightweight linux distro
    python_version="3.10"  # we add a recent Python version
).pip_install(  # and we install the following packages:
    "langchain==0.0.184",
    # 🦜🔗: a framework for building apps with LLMs
    "openai~=0.27.7",
    # high-quality language models and cheap embeddings
    "tiktoken",
    # tokenizer for OpenAI models
    "faiss-cpu",
    # vector storage and similarity search
    "pymongo[srv]==3.11",
    # python client for MongoDB, our data persistence solution
    "gradio~=3.34",
    # simple web UIs in Python, from 🤗
    "gantry==0.5.6",
    # 🏗️: monitoring, observability, and continual improvement for ML systems
)

# we define a Stub to hold all the pieces of our app
# most of the rest of this file just adds features onto this Stub
stub = modal.Stub(
    name="docchat-backend",
    image=image,
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
        modal.Secret.from_name("mongodb-fsdl"),
        modal.Secret.from_name("openai-api-key-fsdl"),
        modal.Secret.from_name("gantry-api-key-fsdl"),
    ],
    mounts=[
        # we make our local modules available to the container
        modal.Mount.from_local_python_packages(
            "vecstore", "docstore", "utils", "prompts", "prompts_trafo"
        )
    ],
)

VECTOR_DIR = vecstore.VECTOR_DIR
vector_storage = modal.NetworkFileSystem.persisted("vector-vol")


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    cpu=8.0,  # use more cpu for vector storage creation
)
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")


@stub.function(image=image)
def drop_docs(collection: str = None, db: str = None):
    """Drops a collection from the document storage."""
    import docstore

    docstore.drop(collection, db)

def prep_documents_for_vector_storage_pdf(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    ids, texts, metadatas = [], [], []
    pretty_log("🦜 DOCUMENT 🦜")
    print(type(documents[0]))   
    print(documents[0])
    for document in documents[:1]:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain.text_splitter import LatexTextSplitter

    text_splitter = LatexTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100, allowed_special="all"
        )

    ids, texts, metadatas = [], [], []
    pretty_log("🦜 DOCUMENT 🦜")
    print(type(documents[0]))   
    print(documents[0])
    i = 0
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas
        i+=1

    return ids, texts, metadatas



@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
def cli(query: str):
    answer = qanda.remote(query, with_logging=True)
    pretty_log("🦜 ANSWER 🦜")
    print(answer)


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
def qanda(query: str, request_id=None, with_logging: bool = True) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chat_models import ChatOpenAI

    import prompts_trafo
    import vecstore

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")

    llm = ChatOpenAI(temperature=0, max_tokens=256) #model_name="gpt",
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts_trafo.main,
        document_variable_name="sources",
    )

    result = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    return answer


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
@modal.web_endpoint(method="POST")
def web(request:dict):
    """Exposes our Q&A chain for queries via a web endpoint.
    The name of this endpoint should match the MODAL_ENDPOINT_NAME environment variable as it is pick-up by the frontend.
    """
    import os

    query = request['query']
    pretty_log(f"Received query: {query}")
    
    request_id = request['request_id'] if 'request_id' in request.keys() else None

    if request_id:
        pretty_log(
        f"handling request with client-provided id: {request_id}"
    )
    else:
        pretty_log(
        f"handling request: {query}"
    )
    answer = qanda.remote(
        query,
        request_id=request_id
    )

    pretty_log('query: {query}\nanswer: {answer}')
    return {"query": query, "answer": answer}




'''
from aiohttp import web

async def handle_requests(request: web.Request) -> web.Response:
    text = 'Hello World!'
    if 'text' in request.query:
        text = "\t".join(request.query.getall("text"))
    try:
        await send_to_slack(channel="#random", text=text)
        return web.json_response(data={'message': 'Done!'})
    except SlackApiError as e:
        return web.json_response(data={'message': f"Failed due to {e.response['error']}"})
'''


### How to handle the slack frontend?
# 1. Create a slack app
# 2. Add a slash command
# 3. Add a request url
# 4. Add a bot user
# 5. Add a bot user token
# 6. Add a bot user token to Modal
# 7. Add a bot user token to Gantry
# 8. Add a bot user token to Slack
# 9. Add a bot user token to the slack app
# 10. Add a bot user token to the slack app
# 11. Add