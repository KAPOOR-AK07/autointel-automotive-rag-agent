"""
AUTOINTEL — Streamlit Web UI
==============================
Run: streamlit run app.py

Professional demo interface showing:
  - Vehicle fleet dashboard in sidebar
  - Chat interface with tool indicator badges
  - Live token counter
  - Agent decision transparency
"""

import sqlite3
import streamlit as st
from agent import run_agent, load_knowledge_base, session_tokens

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AutoIntel Agent",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# LOAD RESOURCES (cached — loads once, reused across queries)
# ============================================================
@st.cache_resource
def get_knowledge_base():
    return load_knowledge_base()

@st.cache_data
def get_fleet_data():
    conn = sqlite3.connect("autointel.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.name, v.type, v.max_bhp, v.max_torque_nm,
               v.weight_kg, v.drag_coefficient,
               ROUND(v.max_bhp * 1000.0 / v.weight_kg, 1) as bhp_per_tonne
        FROM vehicles v
        ORDER BY v.max_bhp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

# ============================================================
# SIDEBAR — VEHICLE FLEET DASHBOARD
# ============================================================
st.sidebar.title("🚗 Fleet Dashboard")
st.sidebar.markdown("---")

try:
    fleet = get_fleet_data()
    for row in fleet:
        name, vtype, bhp, torque, weight, cd, bpt = row
        with st.sidebar.expander(f"**{name}**"):
            col1, col2 = st.columns(2)
            col1.metric("Max BHP", f"{bhp}")
            col2.metric("Max Torque", f"{torque} Nm")
            col1.metric("Weight", f"{weight} kg")
            col2.metric("Cd", f"{cd}")
            st.metric("BHP per Tonne", f"{bpt}", help="Key performance indicator")
            st.caption(f"Type: {vtype}")
except Exception as e:
    st.sidebar.error(f"Load autointel.db first: run setup.py\n{e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Session Tokens**")
st.sidebar.caption(f"Input:  {session_tokens['input']}")
st.sidebar.caption(f"Output: {session_tokens['output']}")

# ============================================================
# MAIN PANEL
# ============================================================
st.title("🚗 AutoIntel — Automotive Intelligence Agent")
st.markdown(
    "Ask anything about our **test vehicle fleet** (live data) "
    "or **automotive engineering concepts** (knowledge base). "
    "The agent decides which source to use automatically."
)

# Example questions as clickable buttons
st.markdown("**Try these:**")
col1, col2, col3 = st.columns(3)
q1 = col1.button("Which vehicle has highest BHP?")
q2 = col2.button("How does a turbocharger work?")
q3 = col3.button("Why do EVs have instant torque?")

col4, col5, col6 = st.columns(3)
q4 = col4.button("Average lateral g-force per vehicle?")
q5 = col5.button("What is aerodynamic drag coefficient?")
q6 = col6.button("What is power-to-weight ratio?")

# Set question from button or text input
preset = None
if q1: preset = "Which vehicle in our fleet has the highest maximum BHP?"
if q2: preset = "How does a turbocharger work and what causes turbo lag?"
if q3: preset = "Why do electric vehicles have instant torque from zero RPM?"
if q4: preset = "What is the average lateral g-force for each vehicle in our fleet?"
if q5: preset = "What is the aerodynamic drag coefficient and how does it affect performance?"
if q6: preset = "What is power to weight ratio and why does it matter?"

st.markdown("---")

# Chat history stored in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input
user_input = st.chat_input("Ask about vehicle data or automotive concepts...")

# Use preset button question OR typed input
question = preset or user_input

if question:
    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                collection, embedder = get_knowledge_base()
                answer, tool_used, in_tok, out_tok = run_agent(
                    question, collection, embedder
                )

                # Tool badge
                if tool_used == "DATA":
                    st.success("📊 Used: Vehicle Telemetry Database (SQL Query)")
                else:
                    st.info("🔍 Used: Automotive Knowledge Base (RAG Search)")

                st.write(answer)

                # Token metrics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Input Tokens", in_tok)
                col_b.metric("Output Tokens", out_tok)
                col_c.metric("Session Total", session_tokens["input"] + session_tokens["output"])

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you ran setup.py first and Ollama is running.")

    # Save to history
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer if "answer" in dir() else "Error"
    })

# Show chat history
if st.session_state.chat_history:
    st.markdown("---")
    with st.expander("📜 Chat History"):
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")
