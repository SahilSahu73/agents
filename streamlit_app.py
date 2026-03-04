import json
import os
from collections.abc import Generator
from typing import Any

import httpx
import streamlit as st


def api_url(path: str) -> str:
    base = str(st.session_state.get("api_base_url", "")).rstrip("/")
    return f"{base}{path}"


def auth_headers() -> dict[str, str]:
    token = str(st.session_state.get("session_token", ""))
    return {"Authorization": f"Bearer {token}"} if token else {}


def append_event(event_type: str, payload: Any) -> None:
    events = st.session_state.get("events", [])
    events.append({"event": event_type, "payload": payload})
    st.session_state["events"] = events


def reset_chat_state() -> None:
    st.session_state["session_id"] = ""
    st.session_state["session_token"] = ""
    st.session_state["messages"] = []
    st.session_state["models"] = []


def logout() -> None:
    st.session_state["user_token"] = ""
    reset_chat_state()
    append_event("auth.logout", {"email": st.session_state.get("email", "")})


def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "api_base_url": os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000/api/v1"),
        "email": "",
        "password": "",
        "user_token": "",
        "session_id": "",
        "session_token": "",
        "messages": [],
        "events": [],
        "models": [],
        "selected_model_label": "",
        "active_page": "auth",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------ API actions ------------------
def login() -> None:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            api_url("/auth/login"),
            data={
                "username": st.session_state["email"],
                "password": st.session_state["password"],
                "grant_type": "password",
            },
        )
    response.raise_for_status()
    body = response.json()
    st.session_state["user_token"] = body["access_token"]
    reset_chat_state()
    append_event("auth.login", {"email": st.session_state["email"]})


def register() -> None:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            api_url("/auth/register"),
            json={"email": st.session_state["email"], "password": st.session_state["password"]},
        )
    response.raise_for_status()
    body = response.json()
    st.session_state["user_token"] = body["token"]["access_token"]
    reset_chat_state()
    append_event("auth.register", {"email": st.session_state["email"], "user_id": body["id"]})


def create_chat_session() -> None:
    user_token = str(st.session_state.get("user_token", ""))
    if not user_token:
        raise RuntimeError("User token is missing. Login or register first.")

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            api_url("/auth/session"),
            headers={"Authorization": f"Bearer {user_token}"},
        )
    response.raise_for_status()
    body = response.json()
    st.session_state["session_id"] = body["session_id"]
    st.session_state["session_token"] = body["token"]["access_token"]
    st.session_state["messages"] = []
    append_event("session.created", {"session_id": st.session_state["session_id"]})


def load_models() -> None:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(api_url("/chatbot/models"))
    response.raise_for_status()
    body = response.json()
    st.session_state["models"] = body.get("models", [])
    append_event("models.loaded", {"count": len(st.session_state["models"])})


def clear_messages() -> None:
    with httpx.Client(timeout=30.0) as client:
        response = client.delete(api_url("/chatbot/messages"), headers=auth_headers())
    response.raise_for_status()
    st.session_state["messages"] = []
    append_event("chat.cleared", {"session_id": st.session_state.get("session_id", "")})


def selected_model() -> tuple[str | None, str | None]:
    label = str(st.session_state.get("selected_model_label", ""))
    if not label or "/" not in label:
        return None, None
    provider, model_name = label.split("/", 1)
    return provider, model_name


def stream_chat() -> Generator[str, None, None]:
    provider, model_name = selected_model()
    payload = {
        "messages": st.session_state.get("messages", []),
        "model_provider": provider,
        "model_name": model_name,
    }
    append_event("chat.request", payload)

    output = ""
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        with client.stream(
            "POST",
            api_url("/chatbot/chat/stream"),
            headers={**auth_headers(), "Accept": "text/event-stream"},
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                append_event(f"sse.{data.get('event', 'chunk')}", data)
                if data.get("event") == "chunk":
                    output += str(data.get("content", ""))
                    yield output


# ------------------ UI ------------------
def render_auth_page() -> None:
    st.header("Authentication")
    st.text_input("API Base URL", key="api_base_url")
    st.text_input("Email", key="email")
    st.text_input("Password", type="password", key="password")

    c1, c2 = st.columns(2)
    if c1.button("Login", use_container_width=True):
        try:
            login()
            st.session_state["active_page"] = "chat"
            st.rerun()
        except Exception as exc:
            st.error(f"Login failed: {exc}")

    if c2.button("Register", use_container_width=True):
        try:
            register()
            st.session_state["active_page"] = "chat"
            st.rerun()
        except Exception as exc:
            st.error(f"Registration failed: {exc}")


def render_chat_sidebar() -> None:
    st.text_input("API Base URL", key="api_base_url")
    st.write(f"Session ID: `{st.session_state.get('session_id') or 'not created'}`")

    if st.button("New Chat Session", use_container_width=True):
        try:
            create_chat_session()
        except Exception as exc:
            st.error(f"Session creation failed: {exc}")

    if st.button("Load Models", use_container_width=True):
        try:
            load_models()
        except Exception as exc:
            st.error(f"Loading models failed: {exc}")

    model_labels = [f"{m['provider']}/{m['name']}" for m in st.session_state.get("models", [])]
    st.selectbox("Model", options=[""] + model_labels, key="selected_model_label")

    if st.button("Clear Chat History", use_container_width=True):
        try:
            clear_messages()
        except Exception as exc:
            st.error(f"Clear chat failed: {exc}")

    if st.button("Logout", use_container_width=True):
        logout()
        st.session_state["active_page"] = "auth"
        st.rerun()


def render_chat_page() -> None:
    st.header("Chat")
    if not st.session_state.get("user_token"):
        st.warning("Please login first.")
        st.session_state["active_page"] = "auth"
        st.rerun()

    with st.sidebar:
        render_chat_sidebar()

    if not st.session_state.get("session_token"):
        st.info("Create a chat session to start chatting.")
        return

    left, right = st.columns([2, 1], gap="large")

    with left:
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Type your message")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            final_text = ""
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Generating response..."):
                        streamed = st.write_stream(stream_chat())
                        if isinstance(streamed, str):
                            final_text = streamed
                        else:
                            final_text = "" if streamed is None else str(streamed)
                except Exception as exc:
                    final_text = f"Request failed: {exc}"
                    st.error(final_text)

            st.session_state["messages"].append({"role": "assistant", "content": final_text})

    with right:
        st.subheader("Runtime Events")
        st.caption("Shows request, streaming events, selected model, and final model.")
        st.json(st.session_state.get("events", [])[-40:])

        if st.session_state.get("models"):
            st.subheader("Model Registry")
            st.dataframe(st.session_state["models"], use_container_width=True)


def main() -> None:
    ensure_state()
    st.set_page_config(page_title="Agents Chat UI", layout="wide")
    st.title("Agents Chat UI")
    st.caption("Streamlit frontend for your FastAPI agent backend")

    # Two-page app flow: unauthenticated users only see Auth, authenticated users can access Chat.
    if not st.session_state.get("user_token"):
        st.session_state["active_page"] = "auth"

    page = st.session_state.get("active_page", "auth")
    if page == "auth":
        render_auth_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
