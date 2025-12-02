#!/usr/bin/env python3
"""
Clean one-shot SQL assistant with:
- SQL-only generation prompt
- Natural language answer prompt
- Intelligent entity resolution
- Real SQL execution
- No context memory
"""

import os
import re
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ----------------------------------------------------------
# 0. Load API Key
# ----------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")


# ----------------------------------------------------------
# 1. DB Config
# ----------------------------------------------------------
DB_URI = "postgresql+psycopg2://postgres:1122@localhost:5432/leads_development"
SCHEMA_FILE = "./schema"


# ----------------------------------------------------------
# 2. Load Rails schema.rb
# ----------------------------------------------------------
def load_schema_text(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()

SCHEMA_TEXT = load_schema_text(SCHEMA_FILE)


# ----------------------------------------------------------
# 3. Parse schema.rb into {table: [(type, col), ...]}
# ----------------------------------------------------------
def parse_schema_tables(schema_text):
    tables = re.findall(
        r'create_table\s+"([^"]+)"[^\n]*do \|t\|(.*?)end',
        schema_text,
        flags=re.DOTALL
    )
    parsed = {}
    for table, body in tables:
        cols = re.findall(r't\.(\w+)\s+"([^"]+)"', body)
        parsed[table] = cols
    return parsed

TABLES = parse_schema_tables(SCHEMA_TEXT)


def build_schema_memory_text(tables):
    lines = ["Database schema overview:"]
    for table, cols in tables.items():
        lines.append(f"\nTABLE: {table}")
        for ctype, col in cols:
            lines.append(f"  - {col}: {ctype}")
    return "\n".join(lines)

SCHEMA_MEMORY = build_schema_memory_text(TABLES)


# ----------------------------------------------------------
# 4. Detect searchable text columns for fuzzy matching
# ----------------------------------------------------------
def detect_searchable_text_columns(tables):
    text_types = {"string", "text", "citext", "varchar", "char"}
    keywords = (
        "name", "full", "first", "last", "email", "owner",
        "contact", "company", "title", "lead", "client"
    )

    searchable = []
    for table, cols in tables.items():
        for ctype, col in cols:
            col_l = col.lower()
            if ctype in text_types or any(k in col_l for k in keywords):
                searchable.append((table, col))
    return sorted(set(searchable))

SEARCHABLE_COLUMNS = detect_searchable_text_columns(TABLES)


# ----------------------------------------------------------
# 5. Safe SQL check (strict)
# ----------------------------------------------------------
def is_safe_sql(sql: str):
    patterns = [
        r"^\s*DROP\s",
        r"^\s*DELETE\s",
        r"^\s*UPDATE\s",
        r"^\s*ALTER\s",
        r"^\s*TRUNCATE\s",
        r"^\s*INSERT\s",
        r"^\s*CREATE\s",
    ]
    up = sql.upper()
    for p in patterns:
        if re.search(p, up, flags=re.MULTILINE):
            return False
    return True


# ----------------------------------------------------------
# 6. DB Engine
# ----------------------------------------------------------
engine = create_engine(DB_URI)


# ----------------------------------------------------------
# 7. LLM
# ----------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# ----------------------------------------------------------
# 8. PROMPTS
# ----------------------------------------------------------

# ---------- SQL GENERATION PROMPT ----------
SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""
You are an expert PostgreSQL query generator.
Your ONLY job is to output a **valid SELECT SQL query**.

RULES:
- Return ONLY SQL. No explanation. No English. No markdown. No backticks.
- Output must be a single SELECT query.
- SQL MUST be safe (no UPDATE, DELETE, INSERT, ALTER, DROP, TRUNCATE, CREATE).
- Use ILIKE for fuzzy text matching.
- Do NOT guess tables or columns not in the schema.

Schema:
{SCHEMA_MEMORY}

Searchable text columns:
{json.dumps(SEARCHABLE_COLUMNS)}
"""),
    ("human", "{question}")
])


# ---------- EXPLANATION PROMPT ----------
EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful SQL analyst.
You receive:
- User question
- Column names
- Actual SQL result rows (sampled)

Your job:
- Provide a short, clear, human-readable answer.
- DO NOT show SQL.
- DO NOT mention SQL.
- DO NOT guess. Base answer only on rows provided.
"""),
    ("human", """
Question: {question}

Columns: {columns}
Rows: {rows}

Write the final answer:
""")
])


# ----------------------------------------------------------
# 9. Keyword extraction
# ----------------------------------------------------------
STOPWORDS = {
    "how","many","did","in","on","the","for","what","is","was","are",
    "this","that","of","a","an","month","year","between","to","from",
}

def extract_keywords(question):
    found = set()

    # quoted strings
    for m in re.findall(r'["\']([^"\']+)["\']', question):
        found.add(m)

    # capitalized words
    for cap in re.findall(r"\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)\b", question):
        if cap.lower() not in STOPWORDS:
            found.add(cap)

    # fallback: long words
    if not found:
        for w in re.findall(r"\b([A-Za-z]{4,})\b", question):
            if w.lower() not in STOPWORDS:
                found.add(w)

    return list(found)


# ----------------------------------------------------------
# 10. Entity Resolution Search
# ----------------------------------------------------------
def run_resolution_search(keyword: str, limit=5):
    matches = []
    kw = f"%{keyword}%"

    for table, col in SEARCHABLE_COLUMNS:
        sql = f"SELECT id, {col} AS v FROM {table} WHERE {col} ILIKE :kw LIMIT {limit}"
        try:
            with engine.connect() as c:
                rows = c.execute(text(sql), {"kw": kw}).fetchall()
                for r in rows:
                    matches.append({
                        "table": table,
                        "column": col,
                        "id": r[0],
                        "value": r[1]
                    })
        except:
            continue

    return matches


# ----------------------------------------------------------
# 11. SQL Generation
# ----------------------------------------------------------
def generate_sql(question, resolution_results):
    summary = "\n".join(
        f"{r['table']}.{r['column']} = '{r['value']}' (id={r['id']})"
        for r in resolution_results
    ) or "No matches found."

    full_prompt = f"""
User question: {question}

Resolved entities:
{summary}

Generate a single safe SELECT SQL query.
Return ONLY SQL.
"""

    chain = SQL_PROMPT | llm
    sql = chain.invoke({"question": full_prompt}).content.strip()

    # Remove ```sql blocks
    m = re.search(r"```sql(.*?)```", sql, flags=re.DOTALL)
    if m:
        sql = m.group(1).strip()

    return sql


# ----------------------------------------------------------
# 12. Execute SQL
# ----------------------------------------------------------
def execute_sql(sql):
    if not is_safe_sql(sql):
        raise ValueError("Unsafe SQL detected.")

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        cols = list(result.keys())
    return rows, cols



def rows_to_jsonable(rows, columns):
    """Convert SQLAlchemy Row objects into plain dicts for JSON serialization."""
    out = []
    for r in rows:
        row_dict = {}
        for idx, col in enumerate(columns):
            val = r[idx]
            # Convert non-serializable objects to strings
            if not isinstance(val, (str, int, float, bool, type(None))):
                val = str(val)
            row_dict[col] = val
        out.append(row_dict)
    return out


# ----------------------------------------------------------
# 13. Natural-language Explanation
# ----------------------------------------------------------
def explain(question, rows, columns):
    safe_rows = rows_to_jsonable(rows[:40], columns)

    chain = EXPLAIN_PROMPT | llm
    return chain.invoke({
        "question": question,
        "columns": json.dumps(columns),
        "rows": json.dumps(safe_rows)
    }).content.strip()



# ----------------------------------------------------------
# 14. Full pipeline
# ----------------------------------------------------------
def answer_question(question):
    keywords = extract_keywords(question)

    resolution_results = []
    for kw in keywords:
        resolution_results += run_resolution_search(kw)

    sql = generate_sql(question, resolution_results)

    try:
        rows, cols = execute_sql(sql)
    except Exception as e:
        return f"❌ SQL Error: {e}"

    return explain(question, rows, cols)


# ----------------------------------------------------------
# 15. CLI
# ----------------------------------------------------------
def main():
    print("🚀 One-shot SQL Assistant Ready")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Goodbye 👋")
            break

        ans = answer_question(q)
        print("\nAI:", ans)
        print("-" * 60)


if __name__ == "__main__":
    main()
