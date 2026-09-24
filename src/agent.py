"""
Standalone GenAI agent for the Retail Demand Forecasting project.

Unlike 06_genai_agent.ipynb (which runs inside Databricks using `spark` and
`dbutils.secrets`), this module runs OUTSIDE Databricks — e.g. locally via
Streamlit — and connects to the Gold Delta table over the Databricks SQL
Connector, using credentials from a local .env file.

Required environment variables (see .env.example):
    OPENAI_API_KEY
    DATABRICKS_SERVER_HOSTNAME
    DATABRICKS_HTTP_PATH
    DATABRICKS_TOKEN
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from databricks import sql as databricks_sql

load_dotenv()

# --- Clients -----------------------------------------------------------

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DATABRICKS_SERVER_HOSTNAME = os.environ["DATABRICKS_SERVER_HOSTNAME"]
DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_HTTP_PATH"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]


def _run_query(query: str):
    """Run a SQL query against the Databricks SQL Warehouse and return rows as dicts."""
    with databricks_sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]


# --- Tools the agent can call -------------------------------------------

def get_historical_trend(store_id: int, days: int = 30):
    """Get recent sales trend statistics for a specific store."""
    rows = _run_query(f"""
        SELECT Date, Sales, Promo, IsStateHoliday
        FROM retail_project.gold.sales_features
        WHERE Store = {store_id}
        ORDER BY Date DESC
        LIMIT {days}
    """)

    if not rows:
        return {"store_id": store_id, "error": "No data found for this store."}

    sales_values = [r["Sales"] for r in rows]
    return {
        "store_id": store_id,
        "avg_sales": round(sum(sales_values) / len(sales_values), 2),
        "min_sales": int(min(sales_values)),
        "max_sales": int(max(sales_values)),
        "days_analyzed": len(rows),
    }


def get_forecast_summary(store_id: int):
    """Get forecasting model accuracy context for a specific store."""
    # Static context reflecting the project's own holdout evaluation results.
    # A fuller production version could call the registered model directly
    # via Databricks Model Serving instead of returning static context.
    return {
        "store_id": store_id,
        "model": "XGBoost (all-store) or Prophet (per-store)",
        "typical_error_pct": "9-10% MAPE based on holdout evaluation",
        "note": "Forecast accuracy varies by store; promotions and weather are significant factors.",
    }


AVAILABLE_FUNCTIONS = {
    "get_historical_trend": get_historical_trend,
    "get_forecast_summary": get_forecast_summary,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_historical_trend",
            "description": "Get recent sales trend statistics for a specific store",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "The store ID number"},
                    "days": {"type": "integer", "description": "Number of recent days to analyze, default 30"},
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_summary",
            "description": "Get forecasting model accuracy context for a specific store",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "integer", "description": "The store ID number"},
                },
                "required": ["store_id"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a retail analytics assistant. You have access to tools that "
    "query real sales data and forecasting model results. Use the tools to "
    "answer questions accurately, and explain the numbers in plain language "
    "for a business stakeholder."
)


def ask_agent(user_question: str, history: list | None = None) -> str:
    """
    Ask the agent a question. `history` (optional) is a list of prior
    {"role": ..., "content": ...} messages for multi-turn conversations.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_response = AVAILABLE_FUNCTIONS[function_name](**function_args)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return second_response.choices[0].message.content

    return response_message.content