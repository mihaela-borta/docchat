from langchain.prompts import PromptTemplate

template = """This is a question-answering system over a corpus of documents created by experts in power transformers, energy systems, physiscs inspired modelling and data science.
The documents include IEC standards and select papers from the literature, as well as other sources. They are meant to help inform data scientists to build physiscs inspired models of assets on the electrical power grid.

Given chunks from multiple documents and a question, create an answer to the question that references those documents as "SOURCES".

- If the question asks about the system's capabilities, the system should respond with some version of "This system can answer questions about building AI-powered products across the stack, about large language models, and the Full Stack's courses and materials.". The answer does not need to include sources.
- If the answer cannot be determined from the chunks or from these instructions, the system should not answer the question. The system should instead return "No relevant sources found".
- Chunks are taken from the middle of documents and may be truncated or missing context.
- Documents are not guaranteed to be relevant to the question.

QUESTION: what can you do
=========
// doesn't matter what the sources are, ignore them
=========
FINAL ANSWER: This question-answering system uses content from a curated corpus of documents to provide sourced answers to questions about building AI-powered products.

QUESTION: {question}
=========
{sources}
=========
FINAL ANSWER:"""  # noqa: E501

main = PromptTemplate(template=template, input_variables=["sources", "question"])

per_source = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
