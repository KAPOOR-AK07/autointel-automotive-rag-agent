"""
AUTOINTEL AGENT — Terminal Version
====================================
Demonstrates:
  - Agentic AI  : Agent decides which tool to use for each question
  - RAG         : Retrieves relevant automotive knowledge from ChromaDB
  - LLM         : Ollama generates natural language answers locally
  - SQL Gen     : LLM writes SQL from plain English questions
  - Token Count : Tracks tokens used per query and session total

Run: python agent.py
"""

import sqlite3
import sys
import re
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

OLLAMA_MODEL = "llama3.2"
TOP_K = 3
session_tokens = {"input": 0, "output": 0}


# ============================================================
# LOAD VECTOR DB + EMBEDDING MODEL
# ============================================================
def load_knowledge_base():
    print("\n Loading automotive knowledge base...", flush=True)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("automotive_knowledge")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(" Knowledge base ready!", flush=True)
    return collection, embedder


# ============================================================
# TOOL 1: SEARCH AUTOMOTIVE KNOWLEDGE DOCS (RAG)
# ============================================================
def tool_search_docs(question, collection, embedder):
    """
    RAG: Embeds the question, finds similar chunks in ChromaDB,
    returns relevant automotive engineering context.
    """
    query_vector = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=TOP_K)
    chunks = results["documents"][0]
    return "\n\n".join(chunks)


# ============================================================
# TOOL 2: QUERY VEHICLE TELEMETRY DATABASE
# ============================================================
def tool_query_database(question):
    """
    Agentic SQL: LLM generates SQL from plain English,
    executes against SQLite telemetry database,
    returns structured results as text.
    """
    schema = """
SQLite tables:
  vehicles(id, name, type, engine, max_bhp, max_torque_nm, weight_kg, drag_coefficient)
  telemetry(id, vehicle_id, timestamp, speed_kmh, rpm, torque_nm, bhp, throttle_pct, brake_pct, lateral_g, engine_temp_c)

Vehicle names: Mahindra XEV 9e, Tata Curvv EV, BMW M3 Competition, Toyota GR86, Maruti Suzuki Swift
"""

    sql_prompt = f"""You are a SQL expert. Write a SQLite SELECT query for this question.

Schema:
{schema}

Question: {question}

Rules:
- Write ONLY the raw SQL query with no explanation
- Do not use markdown or code blocks
- Use JOIN between vehicles and telemetry using vehicle_id
- Use AVG, MAX, MIN for aggregations
- Always include vehicle name in results"""

    sql_response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": sql_prompt}]
    )

    # Track tokens used for SQL generation
    session_tokens["input"] += sql_response.get("prompt_eval_count", 0)
    session_tokens["output"] += sql_response.get("eval_count", 0)

    # Clean any markdown formatting the model might add
    sql = sql_response["message"]["content"].strip()
    sql = re.sub(r"```sql|```", "", sql).strip()

    try:
        conn = sqlite3.connect("autointel.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()

        if not rows:
            return "No data found for that query."

        # Format results as readable text
        header = " | ".join(columns)
        divider = "-" * len(header)
        result_lines = [header, divider]
        for row in rows[:10]:  # Limit to 10 rows for readability
            result_lines.append(" | ".join(str(v) for v in row))

        return "\n".join(result_lines)

    except Exception as e:
        # Fallback query if LLM-generated SQL fails
        conn = sqlite3.connect("autointel.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT v.name, v.max_bhp, v.max_torque_nm,
                   ROUND(AVG(t.speed_kmh),1) as avg_speed_kmh,
                   ROUND(AVG(t.lateral_g),2) as avg_lateral_g,
                   ROUND(AVG(t.engine_temp_c),1) as avg_temp_c
            FROM vehicles v
            JOIN telemetry t ON v.id = t.vehicle_id
            GROUP BY v.id
            ORDER BY v.max_bhp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        lines = ["Vehicle | Max BHP | Max Torque Nm | Avg Speed kmh | Avg G | Avg Temp C", "-"*75]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)


# ============================================================
# AGENT ROUTER — Decides which tool to use
# ============================================================
def decide_tool(question):
    """
    This is the 'Agent' brain.
    It asks the LLM to choose between two tools based on the question.
    Reliable with small models because it's a simple binary choice.
    """
    routing_prompt = f"""You are a routing system for an automotive intelligence tool.

Classify this question into ONE category:
- DATA: questions about specific vehicle numbers, performance figures, comparisons, test results, fleet statistics from our database
- DOCS: questions about concepts, how things work, engineering explanations, definitions

Question: {question}

Reply with ONE WORD only — either DATA or DOCS:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": routing_prompt}]
    )

    session_tokens["input"] += response.get("prompt_eval_count", 0)
    session_tokens["output"] += response.get("eval_count", 0)

    answer = response["message"]["content"].strip().upper()
    return "DATA" if "DATA" in answer else "DOCS"


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================
def run_agent(question, collection, embedder):
    """
    Full agent pipeline:
      1. Decide which tool to use (Agentic routing)
      2. Execute the right tool (RAG or SQL)
      3. Build augmented prompt with retrieved context
      4. Generate final answer with LLM
      5. Count and return tokens used
    """

    # STEP 1: Agent decides tool
    tool_used = decide_tool(question)
    print(f"\n   Agent decision: using {tool_used} tool", flush=True)

    # STEP 2: Execute the chosen tool
    if tool_used == "DOCS":
        print("   Searching automotive knowledge base (RAG)...", flush=True)
        context = tool_search_docs(question, collection, embedder)
        context_label = "Automotive Engineering Knowledge"
    else:
        print("   Querying vehicle telemetry database (SQL)...", flush=True)
        context = tool_query_database(question)
        context_label = "Vehicle Telemetry Data"

    # STEP 3: Build augmented prompt (Augmentation stage)
    final_prompt = f"""You are AutoIntel, an expert automotive data intelligence assistant.
Answer the question using ONLY the context provided below.
Be specific, technical, and cite numbers where available.
If the answer is not in the context, say so clearly.

{context_label}:
{context}

Question: {question}

Answer:"""

    # STEP 4: Generate answer with LLM
    print("   Generating answer with Ollama...", flush=True)
    final_response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": final_prompt}]
    )

    # STEP 5: Track tokens
    input_tokens = final_response.get("prompt_eval_count", 0)
    output_tokens = final_response.get("eval_count", 0)
    session_tokens["input"] += input_tokens
    session_tokens["output"] += output_tokens

    answer = final_response["message"]["content"].strip()

    return answer, tool_used, input_tokens, output_tokens


# ============================================================
# TERMINAL CHAT LOOP
# ============================================================
if __name__ == "__main__":
    collection, embedder = load_knowledge_base()

    print("\n" + "="*55, flush=True)
    print(" AUTOINTEL AGENT READY", flush=True)
    print("="*55, flush=True)
    print(" Try these questions:", flush=True)
    print("   DATA: 'Which vehicle has the highest average BHP?'", flush=True)
    print("   DATA: 'What is the average lateral g-force for BMW M3?'", flush=True)
    print("   DOCS: 'How does a turbocharger work?'", flush=True)
    print("   DOCS: 'What is downforce and how does it affect cornering?'", flush=True)
    print("   DOCS: 'Why do EVs have instant torque?'", flush=True)
    print(" Type 'quit' to exit\n", flush=True)

    while True:
        print("You: ", end="", flush=True)
        try:
            question = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(f"\nSession complete!")
            print(f"Total tokens used — Input: {session_tokens['input']} | Output: {session_tokens['output']}")
            break

        try:
            answer, tool, in_tok, out_tok = run_agent(question, collection, embedder)
            print(f"\n Answer ({tool} tool used):", flush=True)
            print(f"{answer}", flush=True)
            print(f"\n Tokens this query — Input: {in_tok} | Output: {out_tok}", flush=True)
            print(f" Session total    — Input: {session_tokens['input']} | Output: {session_tokens['output']}", flush=True)
            print("-"*55, flush=True)
        except Exception as e:
            print(f"\n Error: {e}", flush=True)
            print(" Make sure setup.py was run and Ollama is running.", flush=True)
