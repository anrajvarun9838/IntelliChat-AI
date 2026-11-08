"""Supabase Auth (email + password) and the shared page bootstrap.

Every page starts with `init_page()`, which (1) configures the page, (2) checks
that required settings exist, (3) requires login, and (4) starts the runtime.
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

LOGIN_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
.stApp {
    background: radial-gradient(1200px 600px at 15% 10%, rgba(99,102,241,.22), transparent 60%),
                radial-gradient(900px 500px at 90% 90%, rgba(236,72,153,.18), transparent 60%);
}
.block-container { padding-top: 7vh; max-width: 480px; }
.brand { text-align: center; margin-bottom: 1.2rem; }
.brand .logo { font-size: 3rem; line-height: 1; }
.brand h1 {
    margin: .3rem 0 0 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -.02em;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.brand p { margin: .3rem 0 0 0; opacity: .7; font-size: .95rem; }
[data-testid="stForm"] {
    border: 1px solid rgba(128,128,128,.25); border-radius: 18px; padding: 1.4rem 1.4rem 1rem 1.4rem;
    background: rgba(128,128,128,.07); backdrop-filter: blur(6px);
    box-shadow: 0 10px 35px rgba(0,0,0,.12);
}
[data-testid="stTextInput"] input { border-radius: 10px; }
[data-testid="stFormSubmitButton"] > button { border-radius: 10px; font-weight: 600; width: 100%; }
.foot { text-align: center; opacity: .55; font-size: .8rem; margin-top: 1.2rem; }
</style>
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
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    # Everything lives in one placeholder, so it can be wiped the moment login succeeds
    # (otherwise Streamlit shows the old login page blurred while the app starts).
    slot = st.empty()

    with slot.container():
        st.markdown(
            '<div class="brand"><div class="logo">💬</div><h1>IntelliChat AI</h1>'
            "<p>Chat, ask your documents, and use AI tools. Your data stays private to your account.</p></div>",
            unsafe_allow_html=True)

        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                try:
                    response = _client().auth.sign_in_with_password({"email": email.strip(), "password": password})
                    _start_session(response.user)
                    slot.empty()  # remove the login page right away
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - show Supabase's own message (wrong password, unconfirmed email...)
                    st.error(getattr(exc, "message", None) or friendly_error(exc))

        with signup_tab:
            with st.form("signup_form"):
                email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
                password = st.text_input("Password (min. 8 characters)", type="password", key="signup_password",
                                         placeholder="At least 8 characters")
                submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted:
                if len(password) < 8:
                    st.error("Use a password with at least 8 characters.")
                else:
                    try:
                        response = _client().auth.sign_up({"email": email.strip(), "password": password})
                        if response.session is not None and response.user is not None:
                            _start_session(response.user)
                            slot.empty()
                            st.rerun()
                        else:
                            st.success("Account created. Check your email to confirm it, then log in.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(getattr(exc, "message", None) or friendly_error(exc))

        st.markdown('<div class="foot">By continuing you agree to use this app responsibly.</div>',
                    unsafe_allow_html=True)


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

    # Show the login page immediately; it only needs Supabase Auth, not the database.
    user = current_user()
    if user is None:
        _login_view()
        st.stop()

    from chatbot import get_runtime  # heavy start-up happens only after login

    try:
        with st.spinner("Starting IntelliChat AI…"):
            get_runtime()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Startup failed")
        st.error(f"Could not start the application: {friendly_error(exc)}")
        st.caption("Check DATABASE_URL (use the Supabase *Session pooler* string) and that the project is running.")
        st.stop()

    if "prefs" not in st.session_state:
        db.upsert_profile(user["id"], user["email"])
        st.session_state["prefs"] = db.get_preferences(user["id"])
    return user