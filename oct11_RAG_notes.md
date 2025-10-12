Embeddings

Search based on embedding efficiently: vector stores

Prompting:
    - https://platform.openai.com/docs/guides/prompt-engineering

Chain of thought

--

Redacting the PII data
- Local model remove the PII data

LLM -> Classification (10,000 types of tokens)
LLM_modified = LLM - last layer + binary classification


---
# RAG
Documents -> embeddings -> Vector Store.

User: Question -> question_embedding -> vector_store.find(question_embeddings) -> relavant documents

Give the document answer users question.
Question: {question}
Documents:
{Document Content}
+ relavant documents

--> 
LLM 

