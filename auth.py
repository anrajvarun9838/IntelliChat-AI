"""Supabase Auth (email + password) and the shared page bootstrap.

Every page starts with `init_page()`, which (1) configures the page, (2) checks
that required settings exist, (3) starts the runtime, and (4) requires login.
Each browser session gets its own Supabase auth client, so sessions never mix.
"""
from __future__ import annotations

import logging

import streamlit as st

import database as db
from config import settings
from utils import friendly_error

logger = logging.getLogger("intellichat.auth")

SETUP_HELP = """
### Setup needed
IntelliChat AI is missing required settings. Add the values below to a `.env` file
(local) or to **Streamlit Secrets** (Community Cloud). See `.env.example` and `README.md`.
"""


def _client():
    """Per-session Supabase client (never cached globally: it holds this user's session)."""
    from supabase import create_client

    if "sb_auth" not in st.session_state:
        st.session_state["sb_auth"] = create_client(settings.supabase_url, settings.supabase_anon_key)
    return st.session_state["sb_auth"]


def current_user() -> dict | None:
    return st.session_state.get("user")


def _start_session(user) -> None:
    st.session_state["user"] = {"id": str(user.id), "email": user.email}
    st.session_state.pop("prefs", None)


def _login_view() -> None:
    st.title("💬 IntelliChat AI")
    st.caption("Sign in to continue. Your chats and documents are private to your account.")
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            try:
                response = _client().auth.sign_in_with_password({"email": email.strip(), "password": password})
                _start_session(response.user)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - show Supabase's own message (wrong password, unconfirmed email...)
                st.error(getattr(exc, "message", None) or friendly_error(exc))

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password (min. 8 characters)", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            if len(password) < 8:
                st.error("Use a password with at least 8 characters.")
            else:
                try:
                    response = _client().auth.sign_up({"email": email.strip(), "password": password})
                    if response.session is not None and response.user is not None:
                        _start_session(response.user)
                        st.rerun()
                    else:
                        st.success("Account created. Check your email to confirm it, then log in.")
                except Exception as exc:  # noqa: BLE001
                    st.error(getattr(exc, "message", None) or friendly_error(exc))


def logout() -> None:
    try:
        _client().auth.sign_out()
    except Exception:  # noqa: BLE001 - local logout must always succeed
        logger.warning("Supabase sign-out failed", exc_info=True)
    st.session_state.clear()
    st.rerun()


def init_page(title: str = "IntelliChat AI") -> dict:
    """Common start for every page. Returns the logged-in user ({'id', 'email'})."""
    st.set_page_config(page_title=title, page_icon="💬", layout="wide", initial_sidebar_state="expanded")

    missing = settings.missing()
    if missing:
        st.markdown(SETUP_HELP)
        st.code("\n".join(f"{name}=" for name in missing), language="bash")
        st.stop()

    from chatbot import get_runtime  # imported here so the setup screen works without heavy imports

    try:
        get_runtime()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Startup failed")
        st.error(f"Could not start the application: {friendly_error(exc)}")
        st.caption("Check DATABASE_URL (use the Supabase *Session pooler* string) and that the project is running.")
        st.stop()

    user = current_user()
    if user is None:
        _login_view()
        st.stop()

    if "prefs" not in st.session_state:
        db.upsert_profile(user["id"], user["email"])
        st.session_state["prefs"] = db.get_preferences(user["id"])
    return user
