# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import logging

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, ConfigDict, SecretStr

from . import auth

logger = logging.getLogger(__name__)


class AuthBase(BaseModel, abc.ABC):
    """Base class for authentication implementations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def validate_auth(self):
        pass


class OpenAILLMGatewayAuthBase(AuthBase):
    """Performs OpenAI API proxy token fetch for the LLM Gateway service."""

    cred_dict: dict | None = None
    openai_client: object | None = None  # OpenAI client generation
    openai_async_client: object | None = None  # AsyncOpenAI client generation
    token_data: dict | None = None

    def validate_auth(self):
        """Validates and refreshes the auth token as needed."""
        self.token_data = auth.validate_token_data(self.token_data)
        if not self.token_data:
            self.token_data = auth.get_oauth_token_data(cred_dict=self.cred_dict)
        if not self.token_data:
            logger.error("Failed to acquire token data!")
            return

        _api_key = self.token_data.get("access_token")
        if self.openai_client:
            self.openai_client.api_key = _api_key
        if self.openai_async_client:
            self.openai_async_client.api_key = _api_key


class AuthClientWrapper:
    """Wrapper to inject auth validation before client method calls."""

    def __init__(self, client, auth_base):
        self._client = client
        self._auth_base = auth_base

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr):

            def wrapped(*args, **kwargs):
                self._auth_base.validate_auth()
                return attr(*args, **kwargs)

            return wrapped
        return attr


class AsyncAuthClientWrapper(AuthClientWrapper):
    """Wrapper for async client methods."""

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr):

            async def wrapped(*args, **kwargs):
                self._auth_base.validate_auth()
                return await attr(*args, **kwargs)

            return wrapped
        return attr


class AzureChatOpenAI_LLMGateway(AzureChatOpenAI):
    """
    Wrapper over `AzureChatOpenAI` that implements LLM Gateway authentication.

    You must provide either `cred_dict` with all the credentials, or `cred` fields
    combined with either `env_prefix` or `cred_file_url` to fetch `cred_dict` from.
    """

    cred_dict: dict | None = None
    cred_fields: list[str] | None = None
    env_prefix: str | None = None
    cred_file_url: str | None = None
    auth_base: type[AuthBase] | None = None
    openai_api_key: SecretStr | None = None

    def __init__(self, *args, **kwargs):
        # Set initial placeholder API key if none provided
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = "sk-placeholder"

        # Convert string api key to SecretStr if provided
        if isinstance(kwargs["openai_api_key"], str):
            kwargs["openai_api_key"] = SecretStr(kwargs["openai_api_key"])

        super().__init__(*args, **kwargs)

        # Initialize auth
        if not self.cred_dict:
            self.cred_dict = auth.get_credentials(
                cred_fields=self.cred_fields,
                env_prefix=self.env_prefix,
                file_url=self.cred_file_url,
            )
        self.auth_base = OpenAILLMGatewayAuthBase(cred_dict=self.cred_dict)

        # Set up auth-wrapped clients
        # What the LangChain OpenAI code calls `self.client` is
        # actually a COMPLETION and not a CLIENT; this is why
        # just `self.client.api_key=<...>` is not working, naturally.
        self.auth_base.openai_client = self.client._client
        self.auth_base.openai_async_client = self.async_client._client

        # Replace the completion clients with wrapped versions
        self.client = AuthClientWrapper(self.client, self.auth_base)
        self.async_client = AsyncAuthClientWrapper(self.async_client, self.auth_base)


class AzureOpenAIEmbeddings_LLMGateway(AzureOpenAIEmbeddings):
    """
    Wrapper over `AzureOpenAIEmbeddings` that implements LLM Gateway authentication.

    You must provide either `cred_dict` with all the credentials, or `cred` fields
    combined with either `env_prefix` or `cred_file_url` to fetch `cred_dict` from.
    """

    cred_dict: dict | None = None
    cred_fields: list[str] | None = None
    env_prefix: str | None = None
    cred_file_url: str | None = None
    auth_base: type[AuthBase] | None = None
    openai_api_key: SecretStr | None = None

    def __init__(self, *args, **kwargs):
        # Set initial placeholder API key if none provided
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = "sk-placeholder"

        # Convert string api key to SecretStr if provided
        if isinstance(kwargs["openai_api_key"], str):
            kwargs["openai_api_key"] = SecretStr(kwargs["openai_api_key"])

        super().__init__(*args, **kwargs)

        # Initialize auth
        if not self.cred_dict:
            self.cred_dict = auth.get_credentials(
                cred_fields=self.cred_fields,
                env_prefix=self.env_prefix,
                file_url=self.cred_file_url,
            )
        self.auth_base = OpenAILLMGatewayAuthBase(cred_dict=self.cred_dict)

        # Set up auth-wrapped clients
        # What the LangChain OpenAI code calls `self.client` is
        # actually an EMBEDDING and not a CLIENT; this is why
        # just `self.client.api_key=<...>` is not working, naturally.
        self.auth_base.openai_client = self.client._client
        self.auth_base.openai_async_client = self.async_client._client

        # Replace the embedding clients with wrapped versions
        self.client = AuthClientWrapper(self.client, self.auth_base)
        self.async_client = AsyncAuthClientWrapper(self.async_client, self.auth_base)
