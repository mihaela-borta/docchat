from langchain.prompts import PromptTemplate

template = """This is a question-answering system over a corpus of documents created by experts in sound and music computing.
The documents include select papers from the literature which cover theoretical and practical knowledge about signal processing, and specific sound technologies and applications, as well as human perception of sound and music.
They are part of the required reading for a Masters in Sound and Music computing.
The person studying this material has a solid enginnering background and has studied signal processing 15 years ago as part of their bachelor, but they have not worked with it since.
Also, the student is has a particular interest in the intersection of AI and sound, and is interested in using AI to create products that are useful for sound and music computing.
The student is particularly interested in the impact of sound perception and would like to use this opportunity to prospect potential insustrial applications of sound and music, especially health and medical focused applications.

Given a question, and a set of potentially relevant documents, create an answer to the question that references those documents as "SOURCES". 
If the documents are relevant to the question, the answer should always include the SOURCES section.
To create the SOURCES section, from the set of relevant documents (not all documents provided may be relevant), list the title and page of the document, in order of relevance.
They are usually in this format:
Source: <Some Author - Some Title>: [<page number>.mmd]

- If the question asks about the system's capabilities, the system should respond with some version of "This system can answer questions about building AI-powered products across the stack, about large language models, and the Full Stack's courses and materials.". The answer does not need to include sources.
- If the answer cannot be determined from the chunks or from these instructions, the system should not answer the question. The system should instead return "No relevant sources found".
- Chunks are taken from the middle of documents and may be truncated or missing context.
- Documents are not guaranteed to be relevant to the question.

QUESTION: what can you do
=========
// doesn't matter what the sources are, ignore them
=========
FINAL ANSWER: This question-answering system uses content from a curated corpus of documents to provide sourced answers to questions about sound and music computing.

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
