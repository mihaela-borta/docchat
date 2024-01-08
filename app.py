"""Builds a CLI, Webhook, and Gradio app for Q&A on the Full Stack corpus.

For details on corpus construction, see the accompanying notebook."""
import modal
from pathlib import Path

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
    # 🏗️: monitoring, observability, and continual improvement for ML systems
    "pandas==2.1.4",
    "google-cloud-bigquery==3.10",
    "google-auth==2.17.3",
    "db-dtypes==1.2.0",
    "tomli==2.0.1",
    "tzlocal==5.2",
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
        modal.Secret.from_name("bigquery_dataset"),
    ],
    mounts=[
        # we make our local modules available to the container
        modal.Mount.from_local_python_packages(
            "vecstore", "docstore", "utils", "prompts",
            "analytics_db.bigquery_utils", "analytics_db.table_schemas",
        ),
        modal.Mount.from_local_dir("./config", remote_path="/root/config"),
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
    for document in documents:
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
            chunk_size=250, chunk_overlap=100, allowed_special="all"
        )

    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas
        
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
def qanda(query: str, with_logging: bool = True) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chains import LLMCheckerChain, SimpleSequentialChain
    from langchain.chat_models import ChatOpenAI

    import prompts
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
    sources_and_scores = vector_index.similarity_search_with_score(query, k=1)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")

    llm = ChatOpenAI(temperature=0.2, max_tokens=256) #model_name="gpt", #initially: 0
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

    result = qa_chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    review_chain = LLMCheckerChain.from_llm(llm, verbose=True)

    overall_chain = SimpleSequentialChain(
        chains=[qa_chain, review_chain], verbose=True
    )

    #checker_chain.run(text)

    return answer, sources, scores


def get_current_date_time():
    from datetime import datetime
    from tzlocal import get_localzone
    import pytz    
    import pandas as pd
    crt_time = datetime.now(get_localzone())
    crt_time = crt_time.astimezone(pytz.utc)
    crt_date = crt_time.date()
    crt_time = pd.Timestamp(crt_time.strftime("%Y-%m-%d %H:%M:%S"))
    return crt_date, crt_time


@stub.function(
    image=image,
)
def log_qna(message_id, query, sources, scores, answer):
    import pandas as pd
    from analytics_db import table_schemas
    from analytics_db.bigquery_utils import BigQueryConnector
    
    bq = BigQueryConnector()
    
    crt_date, crt_time = get_current_date_time()
    
    sources_data = []
    for source, score in zip(sources, scores):
        source_data = {
            "content": source.page_content,  # Replace 'content' with your actual source data
            "title": source.metadata['title'],
            "page": int(source.metadata['page'].replace('.mmd', '')),
            "score": score
        }
        sources_data.append(source_data)

    df = pd.DataFrame({'ts': [crt_time],
            'day': [crt_date],
            'message_id': [message_id],
            'query': [query],
            'sources': [str(sources_data)],
            'answer': [answer]})

    table = "qa_logs"
    job_duration, rows = bq.save_df_to_table(df, table, schema=getattr(table_schemas, table))
    pretty_log(f"Inserted: {rows} rows in {table} in {job_duration} s")
    

@stub.function(
    image=image,
)
def log_frontend_query_id(channel_id, user, message_id):
    import pandas as pd
    from analytics_db import table_schemas
    from analytics_db.bigquery_utils import BigQueryConnector

    bq = BigQueryConnector()

    crt_date, crt_time = get_current_date_time()

    df = pd.DataFrame({'ts': [crt_time],
            'day': [crt_date],
            'channel_id': [channel_id],
            'user': [user],
            'message_id': [message_id]})

    table = "frontend_query_ids"
    job_duration, rows = bq.save_df_to_table(df, table, schema=getattr(table_schemas, table))
    pretty_log(f"Inserted: {rows} rows in {table} in {job_duration} s")


@stub.function(
    image=image,
)
def log_frontend_query_answer_ids(channel_id, query_id, answer_id):
    import pandas as pd
    from analytics_db import table_schemas
    from analytics_db.bigquery_utils import BigQueryConnector

    bq = BigQueryConnector()

    crt_date, crt_time = get_current_date_time()

    df = pd.DataFrame({'ts': [crt_time],
            'day': [crt_date],
            'channel_id': [channel_id],
            'query_id': [query_id],
            'answer_id': [answer_id]})

    table = "frontend_qa_ids"
    job_duration, rows = bq.save_df_to_table(df, table, schema=getattr(table_schemas, table))
    pretty_log(f"Inserted: {rows} rows in {table} in {job_duration} s")


@stub.function(
    image=image,
)
def log_reaction(channel_id, message_id, emoji, count):
    import pandas as pd
    from analytics_db import table_schemas
    from analytics_db.bigquery_utils import BigQueryConnector

    bq = BigQueryConnector()

    table = 'reactions'
    dataset = stub.key_value_store['bq_dataset']

    sql = f"""
    SELECT * FROM {dataset}.{table} WHERE channel_id = '{channel_id}' AND message_id = '{message_id}'
    """
    df = bq.run_query(sql)
    if df is not None and len(df) > 0:
        pretty_log(f"Reaction for message {message_id} is already logged in the DB")
        return

    crt_date, crt_time = get_current_date_time()

    df = pd.DataFrame({'ts': [crt_time],
            'day': [crt_date],
            'channel_id': [channel_id],
            'message_id': [message_id],
            'emoji': [emoji],
            'count': [count]})
    

    job_duration, rows = bq.save_df_to_table(df, table, schema=getattr(table_schemas, table))
    pretty_log(f"Inserted: {rows} rows in {table} in {job_duration} s")


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
    
    if 'request_type' in request.keys() and request['request_type'] == 'query':
        answer, sources, scores = process_query_request(request)
        query = request['query']
        message_id = request['message_id']
        log_qna.remote(message_id=message_id, query=query, sources=sources, scores=scores, answer=answer)
        return answer
    
    elif 'request_type' in request.keys() and request['request_type'] == 'log_query':
        channel_id, user, message_id = request["channel_id"], request["user"], request["message_id"]
        log_frontend_query_id.remote(channel_id, user, message_id)

    elif 'request_type' in request.keys() and request['request_type'] == 'log_query_answer':
        channel_id, query_id, answer_id = request["channel_id"], request["query_id"], request["answer_id"]
        log_frontend_query_answer_ids.remote(channel_id, query_id, answer_id)

    elif 'request_type' in request.keys() and request['request_type'] == 'log_reaction':
        channel_id, message_id = request["channel_id"], request["message_id"]
        for reaction in request['reactions']:
            emoji, count = reaction['emoji'], reaction['count']
            log_reaction.remote(channel_id, message_id, emoji, count)


def process_query_request(request):
    import pandas as pd
    query = request['query']
    pretty_log(f"Received query: {query}")

    request_id = request['request_id'] if 'request_id' in request.keys() else None

    if request_id:
        pretty_log(f"handling request with client-provided id: {request_id}")
    else:
        pretty_log(f"handling request: {query}")
    
    answer, sources, scores = qanda.remote(query)

    if answer == 'No relevant sources found.':
        df = pd.read_csv('data/abbreviations.csv')
        def find_matching_keywords(text, keywords_df):
            keywords_df['match'] = keywords_df['abbreviations'].apply(lambda x: x in text)
        find_matching_keywords(query, df)
        closest_match = df[df['match']==True].groupby(by=['file']).agg(list).reset_index()[['file', 'page', 'abbreviations']].head(100)
        pretty_log(f'query: {query}\nanswer: {answer}\nclosest_match: {closest_match}')

    else:
        pretty_log(f'query: {query}\nanswer: {answer}\nscores:{scores}')
    return answer, sources, scores

