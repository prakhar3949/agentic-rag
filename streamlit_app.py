"""Phase 9's thin UI: one page, one query box, calls `rag/api.py`'s FastAPI backend.

    uv run streamlit run streamlit_app.py

No auth (ROADMAP's explicit cut list) - the "user ID" field below is a plain, self-declared
string, not a login: same ID typed twice gets the same conversation back, nothing more. History
lives in `rag/api.py`'s in-memory `_histories` dict, so it's shared across anyone hitting the same
API with the same ID, and cleared entirely if the API process restarts.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("AGENTIC_RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic Multimodal RAG")
st.caption("Indexed corpus: ai / cs.CL / finance (arXiv)")

if "transcript" not in st.session_state:
    st.session_state.transcript = []  # this browser tab's view of the conversation so far

id_col, reset_col = st.columns([4, 1])
user_id = id_col.text_input(
    "Your name/ID (same ID picks up the same conversation - not a password):",
    value=st.session_state.get("user_id", ""),
) or ""
st.session_state.user_id = user_id
reset_col.write("")
reset_col.write("")
if reset_col.button("New conversation") and user_id.strip():
    try:
        requests.post(f"{API_URL}/reset", json={"user_id": user_id}, timeout=10)
    except requests.RequestException:
        pass
    st.session_state.transcript = []
    st.rerun()

question = st.text_input("Ask a question:") or ""
ask = st.button("Ask", type="primary")

if ask and not user_id.strip():
    st.warning("Enter a name/ID first.")
elif ask and not question.strip():
    st.warning("Type a question first.")
elif ask:
    with st.spinner("Running the agentic pipeline..."):
        try:
            resp = requests.post(
                f"{API_URL}/query", json={"user_id": user_id, "question": question}, timeout=180
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            st.error(f"API request failed - is `rag.api` running at {API_URL}? ({exc})")
            st.stop()
    st.session_state.transcript.append({"raw_question": question, "data": data})

if not st.session_state.transcript:
    st.info("Ask a question to get started.")
else:
    if len(st.session_state.transcript) > 1:
        with st.expander(f"Earlier in this conversation ({len(st.session_state.transcript) - 1} turn(s))"):
            for t in st.session_state.transcript[:-1]:
                st.markdown(f"**You:** {t['raw_question']}")
                st.markdown(f"**Assistant:** {t['data']['answer']}")
                st.divider()

    latest = st.session_state.transcript[-1]
    data = latest["data"]
    if data["question"] != latest["raw_question"]:
        st.caption(f"Turn {data['turn']} — interpreted as: *{data['question']}*")
    else:
        st.caption(f"Turn {data['turn']}")

    if data["blocked_reason"]:
        st.warning(f"**Blocked** ({data['blocked_reason']}): {data['answer']}")
    elif data["needs_clarification"]:
        st.info(f"**Clarification needed:** {data['answer']}")
    else:
        st.subheader("Answer")
        st.write(data["answer"])

        meta_cols = st.columns(3)
        meta_cols[0].caption(f"Route: {data['route']}")
        meta_cols[1].caption(f"Model: {data['model']}  |  {data['latency_s']:.1f}s")
        if data["grounding_score"] is not None:
            meta_cols[2].caption(f"Grounding: {data['grounding_score']:.2f}")
        if data["abstained"]:
            st.info("The model abstained - the retrieved context didn't support an answer.")

        if data["sources"]:
            st.subheader("Citations")
            for s in data["sources"]:
                st.markdown(f"**[{s['handle']}]** {s['provenance']} — score={s['score']:.4f}")
                if s["image_data_uri"]:
                    st.image(s["image_data_uri"], width=300)

        if data["web_sources"]:
            st.subheader("Web sources")
            for w in data["web_sources"]:
                st.markdown(f"**[W{w['handle']}]** [{w['title']}]({w['url']})")

        with st.expander(f"All retrieved chunks ({len(data['all_chunks'])})"):
            for c in data["all_chunks"]:
                st.markdown(f"`{c['score']:.4f}`  {c['provenance']}")
                if c["image_data_uri"]:
                    st.image(c["image_data_uri"], width=200)
