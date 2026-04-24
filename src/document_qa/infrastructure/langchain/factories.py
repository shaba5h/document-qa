from langchain_core.language_models import BaseChatModel

from document_qa.settings import ChatModelSettings


def build_langchain_chat_model(chat_model_settings: ChatModelSettings) -> BaseChatModel:
    match chat_model_settings.provider:
        case "openrouter":
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(
                openrouter_api_key=chat_model_settings.api_key,
                model=chat_model_settings.model_name,
                temperature=chat_model_settings.temperature,
            )

        case _:
            raise ValueError(f"Unsupported provider: {chat_model_settings.provider}")
