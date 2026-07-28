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

from utils.ui_helpers import (
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
    st.caption("Connectivity status")
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

def render_agentic_result(state: dict, heading: str) -> None:
    st.subheader(heading)

    # Verification Badge
    verif = state.get("verification")
    if verif is None:
        v_passed = False
        v_score = 0.0
        v_reasoning = "Verification pending or unexecuted."
        v_evidence = []
        v_unsupported = []
        is_pending = True
    else:
        v_passed = verif.get("passed", False)
        v_score = verif.get("score", 0.0)
        v_reasoning = verif.get("reasoning", "")
        v_evidence = verif.get("evidence_quotes", [])
        v_unsupported = verif.get("unsupported_claims", [])
        is_pending = False

    v_col1, v_col2 = st.columns([1, 2])
    with v_col1:
        if is_pending:
            st.warning("🟡 **Responder Gate: PENDING / UNEXECUTED**")
        elif v_passed:
            st.success(f"🟢 **Responder Gate: VERIFIED** (Score: {v_score:.0%})")
        else:
            st.error(f"🔴 **Responder Gate: UNGROUNDED** (Score: {v_score:.0%})")
    with v_col2:
        st.caption(f"**Verification Reasoning:** {v_reasoning}")

    if v_evidence:
        with st.expander("🔍 Source Evidence Quotes"):
            for q in v_evidence:
                st.markdown(f"- *\"{q}\"*")

    if v_unsupported:
        with st.expander("⚠️ Unsupported / Hallucinated Claims Flagged"):
            for u in v_unsupported:
                st.markdown(f"- ❌ {u}")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Category", (state.get("category") or "Unknown").replace("_", " ").title())
    m2.metric("Doc Type", (state.get("document_type") or "Unknown").replace("_", " ").title())
    m3.metric("Total Tokens", f"{state.get('total_tokens', 0):,}")
    m4.metric("Total Elapsed", f"{state.get('total_elapsed', 0.0):.2f}s")

    # Fixed & Open Schema & Chunks Tabs
    s_tab1, s_tab2, s_tab3, s_tab4, s_tab5 = st.tabs([
        "📋 Fixed Schema", "⚡ Open Schema (Discovered)", "🐾 Planner Trace", "💬 QA Answer", "🧩 Contextual Chunks"
    ])

    with s_tab1:
        st.markdown("#### Fixed Schema Fields")
        fixed = state.get("fixed_schema") or {}
        if isinstance(fixed, dict) and fixed:
            field_rows = format_extraction_fields(fixed)
            st.table({"Field": [r[0] for r in field_rows], "Value": [r[1] for r in field_rows]})
        else:
            st.info("No fixed schema extracted.")

    with s_tab2:
        st.markdown("#### Dynamic Open Schema (Discovered Structure)")
        open_s = state.get("open_schema") or {}
        if isinstance(open_s, dict):
            if open_s.get("summary"):
                st.caption(f"**Structural Summary:** {open_s['summary']}")
            kvs = open_s.get("key_value_pairs") or {}
            if kvs:
                st.markdown("##### Dynamic Key-Value Pairs")
                st.table({"Key": list(kvs.keys()), "Discovered Value": [str(v) for v in kvs.values()]})
            tables = open_s.get("tables") or []
            if tables:
                st.markdown("##### Extracted Product / Claim Tables")
                for tbl in tables:
                    st.write(f"**Table: {tbl.get('table_name', 'Items')}**")
                    st.json(tbl.get("rows", []))
        else:
            st.info("No open schema discovered.")

    with s_tab3:
        st.markdown("#### Planner-Executor Trace")
        history = state.get("history") or []
        if history:
            trace_rows = []
            for step in history:
                act = step.get("action") or {}
                trace_rows.append({
                    "Step": step.get("step_number"),
                    "Tool Action": act.get("tool"),
                    "Rationale": act.get("rationale"),
                    "Result Summary": step.get("result_summary"),
                    "Tokens": step.get("token_cost"),
                    "Time (s)": f"{step.get('inference_time', 0.0):.2f}",
                })
            st.table(trace_rows)
        else:
            st.info("No trace steps available.")

    with s_tab4:
        st.markdown("#### Document QA & Interactive Chatbot")
        filename = state.get("filename", "document.txt")

        if state.get("user_query"):
            st.markdown(f"**Initial Query:** {state['user_query']}")
            st.markdown(f"**Grounded Answer:** {state.get('qa_answer', 'No answer generated.')}")
        else:
            st.info("No initial query was provided during process run.")

        st.divider()
        st.markdown("##### 💬 Continue Chat as Chatbot")
        
        # Initialize session state for interactive chatbot
        chat_key = f"chat_session_{filename}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
            if state.get("user_query") and state.get("qa_answer"):
                st.session_state[chat_key].append({"role": "user", "content": state["user_query"]})
                st.session_state[chat_key].append({"role": "assistant", "content": state["qa_answer"]})

        # Button to launch/toggle active chatbot view
        show_chat = st.checkbox("💬 Enable Interactive Chatbot Interface", value=True, key=f"btn_chat_{filename}")

        if show_chat:
            st.caption(f"Chatting regarding `{filename}` using local ChromaDB vectors & Ollama RAG pipeline.")
            
            # Display chat message thread
            for msg in st.session_state[chat_key]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Chat prompt input
            user_msg = st.chat_input(f"Ask follow-up questions about {filename}...", key=f"input_{filename}")
            if user_msg:
                # Add user message to state and display
                st.session_state[chat_key].append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.write(user_msg)

                # Send RAG chat request to API
                with st.chat_message("assistant"):
                    with st.spinner("Searching ChromaDB vectors & generating response..."):
                        try:
                            payload = {
                                "filename": filename,
                                "query": user_msg,
                                "chat_history": st.session_state[chat_key][:-1],
                            }
                            resp = requests.post(
                                f"{API_BASE_URL}/agentic/chat",
                                json=payload,
                                headers=auth_headers(),
                                timeout=120,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                bot_ans = data["answer"]
                                st.write(bot_ans)
                                st.session_state[chat_key].append({"role": "assistant", "content": bot_ans})
                                
                                if data.get("retrieved_chunks"):
                                    with st.expander("🔍 ChromaDB Retrieved Chunks Used"):
                                        for idx, r_item in enumerate(data["retrieved_chunks"], start=1):
                                            crumbs = r_item.get("metadata", {}).get("breadcrumbs", "Document Section")
                                            score = r_item.get("score", 1.0)
                                            st.markdown(f"**Chunk #{idx} ({crumbs}, score: {score:.2f}):**")
                                            st.caption(r_item.get("document", ""))
                            else:
                                st.error(f"Chat failed ({resp.status_code}): {resp.text}")
                        except requests.RequestException as e:
                            st.error(f"Couldn't reach RAG Chat API: {e}")

    with s_tab5:
        st.markdown("#### 2-Stage Contextual & Structural Chunks")
        struct_chunks = state.get("structured_chunks") or []
        if struct_chunks:
            c1, c2, c3 = st.columns(3)
            tot_raw = sum(c.get("token_count", 0) for c in struct_chunks)
            tot_vec = sum(c.get("vector_token_count", 0) for c in struct_chunks)
            inf_rate = round(((tot_vec - tot_raw) / max(1, tot_raw)) * 100, 2)
            c1.metric("Distinct Chunks", len(struct_chunks))
            c2.metric("Raw -> Vector Tokens", f"{tot_raw} -> {tot_vec}")
            c3.metric("Token Inflation", f"+{inf_rate}%", help="~8% token inflation due to Anthropic contextual retrieval enrichment")

            st.caption("Each chunk stores hierarchical breadcrumbs, metadata tags, and enriched `vector_text` for high-precision retrieval.")

            for idx, chk in enumerate(struct_chunks, start=1):
                meta = chk.get("metadata") or {}
                breadcrumbs = " > ".join(meta.get("section_breadcrumbs", [])) or "Root Document"
                is_tbl = meta.get("is_table", False)
                tbl_badge = "📊 [Table Preserved Intact]" if is_tbl else "📄 [Text Section]"

                with st.expander(f"Chunk #{idx}: {breadcrumbs} {tbl_badge}"):
                    st.markdown(f"**Context Summary:** *{chk.get('context_summary', 'N/A')}*")
                    if meta.get("company") or meta.get("fiscal_year"):
                        st.caption(f"**Metadata:** Company: `{meta.get('company')}`, FY: `{meta.get('fiscal_year')}`")
                    
                    sub_t1, sub_t2 = st.tabs(["Vector Text (Enriched)", "Raw Content"])
                    with sub_t1:
                        st.code(chk.get("vector_text", ""), language="text")
                    with sub_t2:
                        st.code(chk.get("content", ""), language="text")
        else:
            raw_c = state.get("chunks") or []
            st.info(f"Using {len(raw_c)} standard text chunks.")
            for i, c in enumerate(raw_c, start=1):
                with st.expander(f"Chunk #{i}"):
                    st.code(c, language="text")

    with st.expander("Raw AgentState JSON"):
        st.json(state)


# ===== Main UI Tabs =====
tab_classify, tab_agentic, tab_documents = st.tabs([
    "📥 Classify (Deterministic)", "🤖 AgenticIE Router & QA", "📁 My Documents"
])

# ===== Tab 1: classify a new document =====
with tab_classify:
    st.caption("Upload an insurance document to run the deterministic classification pipeline.")

    uploaded_file = st.file_uploader(
        "Choose a document", type=["txt", "md", "pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp"], key="det_uploader"
    )

    if uploaded_file is not None:
        if st.button("Classify document", type="primary", key="det_btn"):
            with st.spinner("Classifying and extracting..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(
                        f"{API_BASE_URL}/classify",
                        files=files,
                        headers=auth_headers(),
                        timeout=600,
                    )
                except requests.exceptions.ReadTimeout:
                    st.error("⏰ Request timed out (600s limit). The local LLM model took too long to complete classification.")
                    resp = None
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


# ===== Tab 2: AgenticIE Router & QA =====
with tab_agentic:
    st.caption("Agentic Information Extraction (Planner, Executor, Responder) with open schema discovery & grounded verification.")

    ag_file = st.file_uploader(
        "Upload document for AgenticIE", type=["txt", "md", "pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp"], key="ag_uploader"
    )
    ag_query = st.text_input("Optional natural language question (QA):", key="ag_query_input")
    use_baseline = st.checkbox("Include deterministic baseline tables/context in AgentState", value=True)

    if ag_file is not None:
        if st.button("Run AgenticIE Engine", type="primary", key="ag_btn"):
            with st.spinner("AgenticIE Planner, Executor & Responder loop running..."):
                try:
                    files = {"file": (ag_file.name, ag_file.getvalue())}
                    params = {}
                    if ag_query.strip():
                        params["user_query"] = ag_query.strip()
                    params["use_deterministic_baseline"] = str(use_baseline).lower()

                    resp = requests.post(
                        f"{API_BASE_URL}/agentic/process",
                        files=files,
                        params=params,
                        headers=auth_headers(),
                        timeout=600,
                    )
                except requests.exceptions.ReadTimeout:
                    st.error("⏰ Request timed out (600s limit). The multi-turn AgenticIE execution took longer than expected on local hardware.")
                    resp = None
                except requests.RequestException as e:
                    st.error(f"Couldn't reach the API: {e}")
                    resp = None

            if resp is not None:
                if resp.status_code == 200:
                    st.session_state.last_agentic_result = resp.json()
                elif resp.status_code == 401:
                    st.error("Session expired -- log in again.")
                else:
                    st.error(f"AgenticIE failed ({resp.status_code}): {resp.text}")

    if st.session_state.get("last_agentic_result"):
        st.divider()
        render_agentic_result(
            st.session_state.last_agentic_result,
            heading=f"AgenticIE Results -- {st.session_state.last_agentic_result.get('filename')}",
        )


# ===== Tab 3: previously uploaded documents =====
with tab_documents:
    st.caption("Documents you've previously classified or processed.")

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