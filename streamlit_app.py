"""
Streamlit UI for the Smart Insurance Document Router.

Talks to the FastAPI backend (app.py) over HTTP -- this file has no
pipeline logic of its own, same "thin wrapper" philosophy as app.py
itself. /classify, /documents, and /documents/{id} all require a
logged-in user now, so the app gates its main content behind login
rather than just optionally attaching a token.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

from ui_helpers import (
    confidence_label,
    format_extraction_fields,
    format_token_summary,
    format_turn_breakdown,
    normalize_document_detail,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Smart Insurance Document Router",
    page_icon="📄",
    layout="wide",
)

# ---------- Session state ----------
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None


def auth_headers() -> dict:
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


# ---------- Shared results renderer -- used for both a fresh classify
# and a previously stored document, via normalize_document_detail() ----------
def render_pipeline_result(result: dict, heading: str) -> None:
    classification = result["classification"]
    extraction = result["extraction"]

    st.subheader(heading)

    col1, col2, col3 = st.columns(3)
    col1.metric("Category", classification["category"].replace("_", " ").title())
    col2.metric("Document Type", classification["document_type"].replace("_", " ").title())
    conf = classification.get("confidence")
    col3.metric(
        "Confidence",
        f"{conf:.0%}" if conf is not None else "unknown",
        confidence_label(conf),
    )
    if classification.get("reasoning"):
        st.caption(f"**Model's reasoning:** {classification['reasoning']}")

    st.markdown("#### Extracted Fields")
    field_rows = format_extraction_fields(extraction["fields"])
    st.table({"Field": [r[0] for r in field_rows], "Value": [r[1] for r in field_rows]})

    st.markdown("#### Token Usage & Cost")
    totals = format_token_summary(result)
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Input Tokens", f"{totals['total_input_tokens']:,}")
    t2.metric("Output Tokens", f"{totals['total_output_tokens']:,}")
    t3.metric("Total Tokens", f"{totals['total_tokens']:,}")
    t4.metric("Inference Time", f"{totals['total_inference_time']:.1f}s")
    t5.metric("Total Pipeline Time", f"{totals['pipeline_elapsed']:.1f}s")

    with st.expander("Per-turn token breakdown"):
        st.table(format_turn_breakdown(result["token_stats"]))

    with st.expander("Raw JSON response"):
        st.json(result)


# ---------- Sidebar: auth (always visible) ----------
with st.sidebar:
    st.header("Account")

    if st.session_state.access_token:
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Log out"):
            st.session_state.access_token = None
            st.session_state.username = None
            st.session_state.pop("last_result", None)
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["Log in", "Register"])

        with tab_login:
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in")
            if submitted:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/token",
                        data={"username": login_username, "password": login_password},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        st.session_state.access_token = resp.json()["access_token"]
                        st.session_state.username = login_username
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Login failed."))
                except requests.RequestException as e:
                    st.error(f"Couldn't reach the API: {e}")

        with tab_register:
            with st.form("register_form"):
                reg_username = st.text_input("Choose a username", key="reg_username")
                reg_password = st.text_input(
                    "Choose a password", type="password", key="reg_password",
                    help="At least 8 characters.",
                )
                reg_submitted = st.form_submit_button("Register")
            if reg_submitted:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/register",
                        json={"username": reg_username, "password": reg_password},
                        timeout=10,
                    )
                    if resp.status_code == 201:
                        st.success("Account created -- you can log in now.")
                    else:
                        st.error(resp.json().get("detail", "Registration failed."))
                except requests.RequestException as e:
                    st.error(f"Couldn't reach the API: {e}")

    st.divider()
    st.caption("Backend status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        if health.get("ollama_reachable"):
            st.caption("🟢 API and Ollama both reachable")
        else:
            st.caption("🟡 API up, but Ollama is not reachable")
    except requests.RequestException:
        st.caption("🔴 API not reachable")


# ---------- Main area ----------
st.title("📄 Smart Insurance Document Router")

if not st.session_state.access_token:
    st.info("Log in or register from the sidebar to classify documents and view your history.")
    st.stop()

tab_classify, tab_documents = st.tabs(["📥 Classify a Document", "📁 My Documents"])

# ===== Tab 1: classify a new document =====
with tab_classify:
    st.caption("Upload an insurance document to classify it and extract structured fields.")

    uploaded_file = st.file_uploader(
        "Choose a document", type=["txt", "md", "pdf", "docx"]
    )

    if uploaded_file is not None:
        if st.button("Classify document", type="primary"):
            with st.spinner("Classifying and extracting -- this can take 20-60s on local hardware..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(
                        f"{API_BASE_URL}/classify",
                        files=files,
                        headers=auth_headers(),
                        timeout=300,
                    )
                except requests.RequestException as e:
                    st.error(f"Couldn't reach the API: {e}")
                    resp = None

            if resp is not None:
                if resp.status_code == 200:
                    st.session_state.last_result = resp.json()
                elif resp.status_code == 400:
                    st.error(f"Unsupported document: {resp.json().get('detail')}")
                elif resp.status_code == 401:
                    st.error("Session expired -- log in again.")
                elif resp.status_code == 502:
                    st.error(f"Classification failed after retries: {resp.json().get('detail')}")
                else:
                    st.error(f"Unexpected error ({resp.status_code}): {resp.text}")

    if st.session_state.get("last_result"):
        st.divider()
        render_pipeline_result(
            st.session_state.last_result,
            heading=f"Results -- {st.session_state.last_result['filename']}",
        )

# ===== Tab 2: previously uploaded documents =====
with tab_documents:
    st.caption("Documents you've previously classified.")

    if st.button("🔄 Refresh list"):
        st.session_state.pop("document_list", None)

    if "document_list" not in st.session_state:
        try:
            resp = requests.get(f"{API_BASE_URL}/documents", headers=auth_headers(), timeout=15)
            if resp.status_code == 200:
                st.session_state.document_list = resp.json()
            elif resp.status_code == 401:
                st.error("Session expired -- log in again.")
                st.session_state.document_list = []
            else:
                st.error(f"Couldn't load documents ({resp.status_code}).")
                st.session_state.document_list = []
        except requests.RequestException as e:
            st.error(f"Couldn't reach the API: {e}")
            st.session_state.document_list = []

    documents = st.session_state.get("document_list", [])

    if not documents:
        st.info("No documents yet -- classify one in the other tab.")
    else:
        st.markdown(f"**{len(documents)} document(s)**")
        st.table([
            {
                "Filename": d["filename"],
                "Category": d["category"].replace("_", " ").title(),
                "Type": d["document_type"].replace("_", " ").title(),
                "Confidence": f"{d['confidence']:.0%}" if d.get("confidence") is not None else "-",
                "Tokens": f"{d['total_tokens']:,}",
                "Uploaded": d["created_at"][:19].replace("T", " "),
            }
            for d in documents
        ])

        options = {f"{d['filename']} (#{d['id']}, {d['created_at'][:19]})": d["id"] for d in documents}
        selected_label = st.selectbox("View details for:", options.keys())

        if selected_label:
            doc_id = options[selected_label]
            try:
                resp = requests.get(
                    f"{API_BASE_URL}/documents/{doc_id}", headers=auth_headers(), timeout=15
                )
                if resp.status_code == 200:
                    st.divider()
                    render_pipeline_result(
                        normalize_document_detail(resp.json()),
                        heading=f"Details -- {resp.json()['filename']}",
                    )
                elif resp.status_code == 404:
                    st.error("Document not found (it may have been removed).")
                else:
                    st.error(f"Couldn't load document details ({resp.status_code}).")
            except requests.RequestException as e:
                st.error(f"Couldn't reach the API: {e}")