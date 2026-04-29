# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


def fill_expiration_time(token_data: dict):
    if token_data and not token_data.get("expires_time"):
        token_data["expires_time"] = time.time() + token_data["expires_in"]


def validate_token_data(token_data: dict | None, min_allowed_expiration=1.0):
    if not token_data:
        return None

    if (
        token_data.get("expires_time", time.time()) - time.time()
        > min_allowed_expiration
    ):
        return token_data
    return None


# TODO: need to generalize to tokens from other providers
def _fetch_token_data(token_url: str, client_id: str, client_secret: str, scope: str):
    """
    Fetch token data from OAuth server.

    Args:
        token_url (str): The url of the OAuth token server.
        client_id (str): The client id for OAuth.
        client_secret (str): The client secret for OAuth.
        scope (str): The scope for OAuth.

    Returns:
        dict: A dictionary with token data if successful, None otherwise.

    Raises:
        Exception: If any error occurs while getting token.
    """
    try:
        data = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope,
            }
        ).encode("utf-8")
        req = urllib.request.Request(token_url, data=data, method="POST")
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        logger.error(f"URL error: {e.reason}")
    except Exception as e:
        logger.error(f"Error occurred while getting OAuth token: {e}")

    return None


def _get_env_creds(prefix: str, values: list[str]) -> dict[str, str] | None:
    cred_dict = {v: os.getenv(f"{prefix}{v}") for v in values}
    for v in cred_dict.values():
        if v is None:
            return None
    return cred_dict


def _get_creds_from_file(file_url: str, values: list[str]) -> dict[str, str] | None:
    try:
        with urllib.request.urlopen(file_url) as response:
            # `response.status` will be `None` when local files are queried (e.g. `file:///c:/creds.json`)
            if response.status == 200 or response.status is None:
                cred_dict = json.loads(response.read().decode("utf-8"))
                cred_dict = {v: cred_dict[v] for v in values}
                for v in cred_dict.values():
                    if v is None:
                        return None
                return cred_dict
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        logger.error(f"URL error: {e.reason}")
    except Exception as e:
        logger.error(
            f"Error occurred while getting fetching credentials file from {file_url}: {e}"
        )
        return None
    return None


def get_credentials(
    cred_fields: list[str], env_prefix: str | None = None, file_url: str | None = None
) -> dict[str, str]:
    """
    Get the SSA credentials.
    """
    cred_dict = None

    # 1) env variables
    if not cred_dict and env_prefix:
        cred_dict = _get_env_creds(prefix=env_prefix, values=cred_fields)
        if cred_dict:
            logger.info(
                f"Found credentials in environment variables prefixed with {env_prefix}"
            )

    # 2) gitlab JSON file
    if not cred_dict and file_url:
        cred_dict = _get_creds_from_file(file_url=file_url, values=cred_fields)
        if cred_dict:
            logger.info(f"Fetched credentials from file at {file_url}")

    return cred_dict


def get_oauth_token_data(cred_dict: dict | None) -> dict | None:
    """
    Get the OAuth token data.

    Args:
        cred_dict (dict | None): Python dictionary, containing credentials necessary for specified auth method.
    Returns:
        dict: The OAuth token data if successful, None otherwise.
    """
    # Check if credentials are available
    if not cred_dict:
        logger.error("No credentials provided for OAuth token fetch")
        return None

    # authenticate for token
    token_data = _fetch_token_data(**cred_dict)
    fill_expiration_time(token_data)

    if not token_data:
        return None

    return token_data


def fetch_creds_and_token_data(
    cred_fields: list[str], env_prefix: str | None, cred_file_url: str | None
) -> dict | None:
    """
    Fetch the credentials from some source first, and then get the OAuth token data based on the creds.

    Args:
        cred_fields (List[str]): List of credential fields to fetch.
        env_prefix (str): Environment variables prefix that will be prepended when attempting to read
            creadentials from env vars (if provided).
        cred_file_url (str): URL of the JSON file containing credentials.
    Returns:
        dict: The OAuth token data if successful, None otherwise.
    """
    # get credentials
    cred_dict = get_credentials(
        cred_fields=cred_fields, env_prefix=env_prefix, file_url=cred_file_url
    )
    return get_oauth_token_data(cred_dict)
