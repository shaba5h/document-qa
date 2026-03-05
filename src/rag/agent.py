from config import Config


def make_vector_store(config: Config):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name=config.huggingface_model,
        model_kwargs={"device": config.huggingface_device},
    )
    return Chroma(
        embedding_function=embeddings,
        collection_name=config.chroma_collection_name,
        persist_directory=str(config.chroma_persist_directory),
    )


def make_agent(config: Config, vector_store):
    from langchain_openrouter import ChatOpenRouter
    from langchain.tools import tool
    from langchain.agents import create_agent

    llm = ChatOpenRouter(
        model=config.openrouter_model,
        temperature=config.openrouter_temperature,
        openrouter_api_key=config.openrouter_api_key.get_secret_value(),
    )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Find relevant information to answer the question."""
        retrieved_docs = vector_store.similarity_search_with_score(query, k=config.retriever_k)
        if config.retriever_score_threshold is not None:
            retrieved_docs = [
                (doc, score) for doc, score in retrieved_docs
                if score <= config.retriever_score_threshold
            ]
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}\nScore: {score}"
            for doc, score in retrieved_docs
        )
        return serialized, retrieved_docs

    return create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=config.agent_system_prompt,
    )
