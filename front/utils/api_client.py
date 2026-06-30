"""Client HTTP vers l'API FastAPI (health, model-info, predict).

Le dashboard ne charge jamais le modèle directement : il appelle l'API,
ce qui reproduit l'architecture cible Front / API / Modèle (cf. API/README.md).
"""

import requests
import streamlit as st

from .config import DEFAULT_API_URL


class APIError(Exception):
    """Erreur de communication avec l'API (réseau, HTTP, payload invalide)."""


def _base_url() -> str:
    url = st.session_state.get("api_url", DEFAULT_API_URL)
    return url.rstrip("/")


def _extract_error(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(data, dict) and "detail" in data:
        return str(data["detail"])
    return str(data)


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_health(url: str) -> dict:
    r = requests.get(f"{url}/health", timeout=3)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_model_info(url: str) -> dict:
    r = requests.get(f"{url}/model-info", timeout=5)
    r.raise_for_status()
    return r.json()


def check_health() -> dict:
    url = _base_url()
    try:
        return _fetch_health(url)
    except requests.RequestException as e:
        raise APIError(f"Connexion impossible à {url} : {e}") from e


def get_model_info() -> dict:
    url = _base_url()
    try:
        return _fetch_model_info(url)
    except requests.RequestException as e:
        raise APIError(f"Connexion impossible à {url} : {e}") from e


def predict(payload: dict, timeout: float = 10) -> dict:
    """Appelle POST /predict avec les 9 features brutes et renvoie le JSON de réponse."""
    url = f"{_base_url()}/predict"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise APIError(f"Connexion impossible à {url} : {e}") from e
    if r.status_code != 200:
        raise APIError(_extract_error(r))
    return r.json()
