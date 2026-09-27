"""
Northbridge Bank - Credit Risk Portfolio Query Engine
Streamlit application

Converted from the Learner Notebook (Project 3: Credit Risk Query Engine).
All pipeline logic (intent classification, SQL generation, validation,
retry, execution, and response generation) is preserved unchanged from the
notebook implementation. Only the input/output layer has been adapted to
Streamlit widgets.
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Northbridge Bank | Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)


# =============================================================================
# Configuration / credential loading
# =============================================================================
def load_credentials():
    """
    Loads OpenAI credentials in the following priority order:
    1. Streamlit secrets (st.secrets) - recommended for Streamlit Cloud deployment
    2. Environment variables (OPENAI_API_KEY / OPENAI_API_BASE)
    3. A local config.json file (same format used in the notebook)

    Returns (api_key, api_base) - api_base may be None (uses OpenAI default).
    """
    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        if "OPENAI_API_BASE" in st.secrets:
            api_base = st.secrets["OPENAI_API_BASE"]
    except Exception:
        pass

    # 2. Environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE")

    # 3. Local config.json (same pattern as the notebook)
    if not api_key and os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            api_key = config.get("OPENAI_API_KEY")
            api_base = config.get("OPENAI_API_BASE")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    return api_key, api_base


# =============================================================================
# Database schema description (passed to the LLM in prompts) - unchanged
# =============================================================================
DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# =============================================================================
# Verified Query Template Library - unchanged SQL from the notebook
# =============================================================================
def build_verified_query_library() -> Dict[str, Dict[str, str]]:
    sql_1 = """
SELECT
  sm.sector_name,
  ROUND(SUM(lm.total_outstanding)/1e6, 2) AS total_outstanding_million,
  ROUND(SUM(CASE
              WHEN lm.asset_classification IN ('Substandard','Doubtful','Loss')
              THEN lm.total_outstanding
              ELSE 0 END)/1e6, 2)
              AS npa_exposure_million
FROM loan_master lm
JOIN sector_master sm USING(sector_code)
GROUP BY sm.sector_name
ORDER BY total_outstanding_million DESC
"""

    sql_2 = """
SELECT
  loan_category,
  ROUND(SUM(total_outstanding)/1e6, 2) AS total_outstanding_million,
  COUNT(DISTINCT loan_account_number) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_million DESC
"""

    sql_3 = """
SELECT
  ifrs9_stage,
  COUNT(DISTINCT loan_account_number) AS loan_count,
  ROUND(SUM(ead_amount)/1e6, 1) AS ead_million,
  ROUND(SUM(ecl_amount)/1e6, 1) AS ecl_million
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

    sql_4 = """
SELECT
    sm.sector_name AS sector,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm USING(loan_account_number)
JOIN sector_master sm USING(sector_code)
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

    sql_5 = """
SELECT
  borrower_name,
  sector_code,
  ROUND(total_outstanding/1e6, 1) total_outstanding_millions,
  asset_classification
FROM loan_master
ORDER BY total_outstanding DESC
LIMIT 10
"""

    sql_6 = """
SELECT
  group_name,
  COUNT(DISTINCT loan_account_number) AS loan_count,
  ROUND(SUM(total_outstanding)/1e6, 1) AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5
"""

    sql_7 = """
SELECT
  loan_account_number,
  borrower_name,
  sector_code,
  ROUND(total_outstanding/1e6, 1) AS total_outstanding_million,
  days_past_due,
  asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""

    sql_8 = """
SELECT
  CASE
    WHEN days_past_due = 0 THEN '0 (Current)'
    WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
    WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
    WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
    ELSE '90+'
  END AS days_past_due,
  COUNT(*) AS loan_count,
  ROUND(SUM(total_outstanding)/1e6, 2) AS total_outstanding_million
FROM loan_master
GROUP BY days_past_due
ORDER BY days_past_due DESC
"""

    sql_9 = """
SELECT
  borrower_id,
  previous_rating,
  internal_rating,
  pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

    sql_10 = """
SELECT
  reporting_date,
  ROUND(SUM(ecl_amount)/1e6, 1) AS ecl_amount_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

    return {
        "VQ1": {
            "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
            "sql": sql_1,
        },
        "VQ2": {
            "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
            "sql": sql_2,
        },
        "VQ3": {
            "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
            "sql": sql_3,
        },
        "VQ4": {
            "description": "Average provision coverage ratio by sector for the latest reporting quarter",
            "sql": sql_4,
        },
        "VQ5": {
            "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
            "sql": sql_5,
        },
        "VQ6": {
            "description": "Top 5 largest exposures aggregated at the business group level",
            "sql": sql_6,
        },
        "VQ7": {
            "description": "All overdue loan accounts with their days past due and asset classification",
            "sql": sql_7,
        },
        "VQ8": {
            "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
            "sql": sql_8,
        },
        "VQ9": {
            "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
            "sql": sql_9,
        },
        "VQ10": {
            "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
            "sql": sql_10,
        },
    }


# =============================================================================
# Pipeline tools - logic preserved exactly from the notebook
# =============================================================================
def classify_intent(user_question, query_library, llm):
    """
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.
    - llm: The bound chat model used for classification.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    """
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""


##Role
You are a credit risk analyzer that decides to classify whether the business user's question can be answered using a query template or needs to generate an SQL.

##Input
#user question
{user_question}

#Content schema
{library_descriptions}

#Instructions
1. Read the user's question very carefully and find the analytical intent.
2. Only use verified if the user’s question is semantically equal to one of the verified query template description, does not have to have exact match.
3. Return the template Id if the template matchers the user question.
4. If none of the templates match, generate a fresh SQL.


### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    """
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.
    - llm: The bound chat model used for generation.

    Returns:
    - str: Candidate SQL query as a string.
    """
    generation_prompt = f"""


#Role
You are a Senior SQLite developer workin in credit risk analytics.

#Input
#user question
{user_question}

#schema context
{schema_context}

#Instructions
1. Write only a single SQL query.
2. Make sure the query is read-only and never try to modify the schema permanently.
3. Ensure all queries are SQLite compatible.
4. Do not hallucinate the column & tables names, use only what is provided in the schema only.
5. No markdown, no explanation or comment is required, generate only single query.

#Output
Return ONLY the SQL query, with no markdown, no explanation or comment
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None):
    """
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - evaluator_llm: The bound chat model used for the relevance check.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    """
    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or "Name" in str(t.ttype)]
    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as", "case",
        "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union", "all", "between", "coalesce",
    }
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
        and not tok.isdigit() and tok not in ("s", "l", "p", "r", "e6")
    ]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage - judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""


# role
You are a SQL relevance evaluator for a credit risk query engine that checks if the SQL query answers the question correctly.

# intent
Your intent is to decide whether the candidate SQL query correctly answers the user's question.

# instruction
1. Does the query have the correct columns and tables?
2. Does the query have the correct filters?
3. Does the query have the correct grouping and ordering?
4. Does the query have the correct aggregation functions?
5. Does the query hander the NPA correctly?

#context
{track_context}

#Input/User question
{user_question}

#canidate SQL
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    """
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.
    - llm: The bound chat model used for regeneration.

    Returns:
    - str: Revised SQL as a string.
    """
    retry_prompt = f"""

#role
You are a Senior SQLite developer fixing a query that failed the validation.

#instructions
1. Fix only the issue causing the validation error.
2. Ensure the query is read-only and never try to modify the schema permanently.
3. Ensure all queries are SQLite compatible.
4. Do not hallucinate the column & tables names, use only what is provided in the schema only.
5. No markdown, no explanation or comment is required, generate only single query.


#Output
Return ONLY the SQL query, with no markdown, no explanation or comment

#input
#user question
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    """
    result = {
        "dataframe": None,
        "reasonable": True,
        "warnings": [],
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    # Reasonableness checks
    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


def generate_response(user_question, dataframe, route, llm, query_id=None):
    """
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - llm: The bound chat model used for narrative generation.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    """
    response_prompt = f"""


#role
You are a credit risk analyst for Northbridge Bank writing the response in plain english for the board commitee.


# instructions
1. Include all the metrics
2. Include amount in millions.
3. Use clear, clean, and professional language for the response.
4. Answer only the question asked, do not give addditonal analysist unless told.
5. Make the response precisie and keep it at 2-4 sentances unless more detail is needed.
7. If data is inconclusive or empty let the user know that clearly.

#input
{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, verbose=False):
    """
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - llm: The bound chat model used for classification/generation/response.
    - evaluator_llm: The bound chat model used for validation relevance checks.
    - verbose (bool): If True, appends intermediate pipeline stages to a trace log.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    """
    trace: List[str] = []

    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "narrative": None,
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    log["route"] = classification["route"]
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")

    trace.append(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    trace.append(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log["candidate_sql"] = candidate_sql

    trace.append(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, log["query_id"])
    log["gate_result"] = gate

    trace.append(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate["passed"]:
        trace.append(f"    Failed check: {gate.get('failed_check')}")
        trace.append(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and log["route"] == "generated":
        trace.append(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context, llm)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        log["gate_result"] = gate

        trace.append(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate["passed"]:
            trace.append(f"    Retry failed check: {gate.get('failed_check')}")
            trace.append(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        trace.append(f"[!] Escalated to human: {gate['details']}")
        return {"log": log, "dataframe": None, "trace": trace, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    log["row_count"] = len(df)

    trace.append(f"[4] Execute: {len(df)} rows returned")
    if exec_result["warnings"]:
        trace.append(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log["route"], llm, log["query_id"])
    log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log["confidence"] = gate.get("relevance_confidence")

    trace.append(f"[6] Response Generation: confidence={log['confidence']}")

    return {"log": log, "dataframe": df, "trace": trace, **log}


# =============================================================================
# Cached resources: DB connection and LLM clients
# =============================================================================
@st.cache_resource
def get_db_connection(db_path: str):
    """Creates a read-only SQLite connection, matching the notebook's approach."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


@st.cache_resource
def get_llms(api_key: str, api_base: Optional[str]):
    """Instantiates the primary and evaluator LLMs used throughout the pipeline."""
    kwargs = {"model": "gpt-4o-mini", "temperature": 0}
    llm = ChatOpenAI(**kwargs)
    evaluator_llm = ChatOpenAI(**kwargs)
    return llm, evaluator_llm


@st.cache_data
def get_verified_query_library():
    return build_verified_query_library()


# =============================================================================
# Streamlit UI
# =============================================================================
def main():
    st.title("🏦 Northbridge Bank - Credit Risk Portfolio Query Engine")
    st.caption(
        "Ask routine commercial lending portfolio questions in plain English. "
        "The engine routes each question to a pre-approved verified query template "
        "or generates and validates a read-only SQL query on the fly."
    )

    api_key, api_base = load_credentials()

    # --- Sidebar: configuration & status ---
    with st.sidebar:
        st.header("Configuration")

        db_path = st.text_input(
            "Database file path",
            value=os.environ.get("DB_PATH", "credit_risk_portfolio.db"),
            help="Path to the read-only SQLite database bundled with the app.",
        )

        if not api_key:
            api_key_input = st.text_input("OpenAI API Key", type="password")
            if api_key_input:
                api_key = api_key_input
                os.environ["OPENAI_API_KEY"] = api_key

        st.divider()

        if not api_key:
            st.error("No OpenAI API key found. Provide one above, or set it via Streamlit secrets / environment variables.")
        else:
            st.success("OpenAI API key loaded.")

        if os.path.exists(db_path):
            st.success(f"Database found: {db_path}")
        else:
            st.error(f"Database not found at: {db_path}")

        st.divider()
        st.subheader("Verified Query Library")
        library = get_verified_query_library()
        for qid, entry in library.items():
            with st.expander(f"{qid}"):
                st.write(entry["description"])
                st.code(entry["sql"].strip(), language="sql")

    if not api_key or not os.path.exists(db_path):
        st.info("Complete the configuration in the sidebar to start using the query engine.")
        return

    conn = get_db_connection(db_path)
    llm, evaluator_llm = get_llms(api_key, api_base)
    verified_query_library = get_verified_query_library()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    tab_query, tab_history, tab_eval = st.tabs(["Ask a Question", "Query History", "Evaluation (Ground Truth)"])

    # -------------------------------------------------------------------
    # Tab 1: Ask a question
    # -------------------------------------------------------------------
    with tab_query:
        with st.form("query_form"):
            user_question = st.text_area(
                "Enter your portfolio question",
                placeholder="e.g. What is the sector-wise NPA breakdown across the portfolio?",
                height=100,
            )
            show_trace = st.checkbox("Show pipeline trace", value=False)
            submitted = st.form_submit_button("Run Query", type="primary")

        if submitted:
            if not user_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running the query engine pipeline..."):
                    try:
                        result = run_pipeline(
                            user_question,
                            conn,
                            verified_query_library,
                            DATABASE_SCHEMA,
                            llm,
                            evaluator_llm,
                            verbose=show_trace,
                        )
                    except Exception as e:
                        st.error(f"Pipeline execution failed: {e}")
                        result = None

                if result is not None:
                    st.session_state["history"].append(result)
                    render_result(result, show_trace, key_suffix=f"current_{len(st.session_state['history'])}")

    # -------------------------------------------------------------------
    # Tab 2: Query history
    # -------------------------------------------------------------------
    with tab_history:
        if not st.session_state["history"]:
            st.info("No queries have been run yet in this session.")
        else:
            for i, result in enumerate(reversed(st.session_state["history"])):
                idx = len(st.session_state["history"]) - i
                with st.expander(f"{idx}. {result['user_question']}"):
                    render_result(result, show_trace=True, key_suffix=f"history_{idx}")

    # -------------------------------------------------------------------
    # Tab 3: Evaluation against ground truth (test_queries.csv)
    # -------------------------------------------------------------------
    with tab_eval:
        st.write(
            "Runs the full pipeline against every case in `test_queries.csv` and reports "
            "Selected Path Accuracy, Selected Query Accuracy, and Average Confidence Score, "
            "matching the notebook's evaluation methodology."
        )
        csv_path = st.text_input("Ground truth CSV path", value="test_queries.csv")

        if st.button("Run Evaluation"):
            if not os.path.exists(csv_path):
                st.error(f"File not found: {csv_path}")
            else:
                ground_truth = pd.read_csv(csv_path)
                evaluation_rows = []
                progress = st.progress(0.0)

                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    tr = run_pipeline(
                        gt["User Query"], conn, verified_query_library, DATABASE_SCHEMA,
                        llm, evaluator_llm, verbose=False,
                    )
                    evaluation_rows.append({
                        "Test Case": gt["Test Case"],
                        "Expected Route": gt["Expected Route"],
                        "Actual Route": tr["route"],
                        "Route Match": tr["route"] == gt["Expected Route"],
                        "Expected Query ID": gt["Expected Query ID"],
                        "Actual Query ID": tr["query_id"],
                        "Query ID Match": (
                            pd.isna(gt["Expected Query ID"]) and pd.isna(tr["query_id"])
                        ) or tr["query_id"] == gt["Expected Query ID"],
                        "Confidence": tr["confidence"],
                        "Rows Returned": tr["row_count"],
                    })
                    progress.progress((i + 1) / len(ground_truth))

                evaluation_df = pd.DataFrame(evaluation_rows)
                path_accuracy = evaluation_df["Route Match"].mean() * 100

                verified_mask = evaluation_df["Expected Route"].str.strip().str.lower() == "verified"
                query_accuracy = evaluation_df.loc[verified_mask, "Query ID Match"].mean() * 100

                numeric_confidence = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
                average_confidence = numeric_confidence.mean()

                col1, col2, col3 = st.columns(3)
                col1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                col2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
                col3.metric("Average Confidence Score", f"{average_confidence:.2f}")

                st.dataframe(evaluation_df, width="stretch")


def render_result(result: Dict[str, Any], show_trace: bool, key_suffix: str = "0"):
    """Renders a single pipeline result: confidence, narrative, SQL, data, trace."""
    confidence = result.get("confidence")
    escalated = result.get("escalated")

    if escalated:
        st.error(f"Escalated to human analyst: {result['gate_result'].get('details')}")
    else:
        route = result.get("route")
        query_id = result.get("query_id")
        badge = f"`{route}`" + (f" · `{query_id}`" if query_id else "")
        st.markdown(f"**Route:** {badge}")

        if isinstance(confidence, (int, float)):
            st.metric("Confidence", f"{confidence:.2f}")
        else:
            st.write(f"**Confidence:** {confidence}")

        st.markdown("#### Narrative")
        st.write(result.get("narrative"))

        st.markdown("#### Executed SQL")
        st.code(result.get("executed_sql", ""), language="sql")

        st.markdown("#### Result Data")
        df = result.get("dataframe")
        if df is not None and not df.empty:
            st.dataframe(df, width="stretch")
            st.download_button(
                "Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"download_{key_suffix}",
            )
        else:
            st.info("No rows returned.")

    if show_trace and result.get("trace"):
        st.markdown("#### Pipeline Trace")
        st.code("\n".join(result["trace"]))


if __name__ == "__main__":
    main()
