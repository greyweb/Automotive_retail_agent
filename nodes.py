import streamlit as st
import sqlite3
import pandas as pd
from typing import List, TypedDict, Annotated, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Import prompts and schema from the prompts file
from prompts import (
    VEHICLE_SCHEMA,
    PLANNER_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    NO_RESULTS_HANDLER_SYSTEM_PROMPT,
    INVENTORY_SUMMARY,
)

# --- Logger setup ---
logger = logging.getLogger("AICarAgent")

# ================================
# 1. AGENT STATE AND LLM
# ================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    sql_query: str
    query_result: str
    final_answer: str
    user_preferences: Dict[str, Any]
    conversation_stage: str
    selected_filters: Dict[str, Any]

# Configuration
GOOGLE_API_KEY = 'AIzaSyDkeo5SHfI_hpqgH2TlRbbvF4VwIQFUi7k' # IMPORTANT: Replace with your actual key or use st.secrets

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

llm = get_llm()

# ================================
# 2. DATABASE AND UTILITY FUNCTIONS
# ================================

@st.cache_data
def run_query(query: str, db_file: str = "car_inventory.db") -> pd.DataFrame:
    """Run a SQL query on a local SQLite database and return the results as a DataFrame."""
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Database query error: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['year', 'make', 'model', 'trim', 'net_price', 'msrp', 'condition', 'body_style', 'fuel_type', 'offers', 'finance_options', 'url'])

def format_conversation_with_filters(messages: List[BaseMessage], filters: Dict[str, Any]) -> str:
    """Format conversation history including filter selections."""
    formatted = []
    
    if filters:
        filter_context = "CUSTOMER FILTER SELECTIONS:\n"
        for key, value in filters.items():
            if value:
                filter_context += f"- {key.replace('_', ' ').title()}: {value}\n"
        formatted.append(filter_context)
    
    for msg in messages:
        speaker = 'Customer' if isinstance(msg, HumanMessage) else 'Assistant'
        formatted.append(f"{speaker}: {msg.content}")
    
    return "\n".join(formatted)

# ================================
# 3. WORKFLOW NODES
# ================================

def planner_node(state: AgentState):
    """UPDATED: Enhanced planner synchronized with actual database structure."""
    try:
        conversation_history = format_conversation_with_filters(
            state["messages"],
            state.get("selected_filters", {})
        )
        
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", "Analyze the conversation and filters to determine the next step.")
        ])

        planner_chain = planner_prompt | llm | StrOutputParser()
        
        response = planner_chain.invoke({
            "schema": VEHICLE_SCHEMA,
            "conversation_history": conversation_history
        })

        return {
            "messages": [AIMessage(content=response)],
            "conversation_stage": "planning" if "GENERATE_SQL" not in response else "ready_to_search"
        }
    except Exception as e:
        logger.error(f"Planner node error: {e}")
        return {
            "messages": [AIMessage(content="I apologize, but I'm having trouble processing your request. Could you please rephrase your question?")],
            "conversation_stage": "error"
        }

def sql_generator_node(state: AgentState):
    """UPDATED: Enhanced SQL generator synchronized with actual column names and data."""
    try:
        conversation_history = format_conversation_with_filters(
            state["messages"],
            state.get("selected_filters", {})
        )
        
        sql_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_GENERATOR_SYSTEM_PROMPT),
            ("human", "Generate the optimized SQL query.")
        ])

        sql_gen_chain = sql_gen_prompt | llm | StrOutputParser()
        
        query = sql_gen_chain.invoke({
            "schema": VEHICLE_SCHEMA,
            "conversation_history": conversation_history
        })
        
        clean_query = query.strip().replace("```sql", "").replace("```", "").strip()
        
        return {"sql_query": clean_query}
    except Exception as e:
        logger.error(f"SQL generator error: {e}")
        return {"sql_query": "SELECT year, make, model, trim, net_price, msrp, condition, body_style, fuel_type, offers, finance_options, url FROM car_details WHERE location = 'Concord' LIMIT 5"}

def sql_executor_node(state: AgentState):
    """Execute SQL with enhanced error handling."""
    query = state["sql_query"]
    
    try:
        result_df = run_query(query)
        
        if result_df.empty:
            return {"query_result": "No vehicles found matching your criteria."}
        
        required_cols = ['make', 'model', 'year', 'net_price', 'url', 'trim', 'msrp', 'condition', 'fuel_type', 'offers', 'finance_options', 'transmission', 'ext_color']
        for col in required_cols:
            if col not in result_df.columns:
                result_df[col] = 'N/A'

        markdown_table = result_df.to_markdown(index=False)
        return {"query_result": markdown_table}
        
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return {"query_result": f"Error executing query: {str(e)}"}

def summarizer_node(state: AgentState):
    """Creates a highly polished summary using the Auto-Genie persona."""
    try:
        conversation_history = format_conversation_with_filters(
            state["messages"],
            state.get("selected_filters", {})
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZER_SYSTEM_PROMPT),
            ("human", "You are Auto-Genie. Please create the beautiful, insightful summary as instructed using the provided search results, including the exterior color and the new call to action format.")
        ])

        summarizer_chain = prompt | llm | StrOutputParser()
        response = summarizer_chain.invoke({
            "query_result": state["query_result"],
            "conversation_history": conversation_history
        })

        return {"final_answer": response}

    except Exception as e:
        logger.error(f"Summarizer error: {e}")
        return {"final_answer": (
            "I'm sorry, I encountered a problem while preparing your personalized recommendations. "
            "Could you please try refining your search or starting over?"
        )}

def no_results_handler_node(state: AgentState):
    """
    Handles the scenario where the SQL query returns no results by providing
    intelligent, context-aware suggestions based on inventory analysis.
    """
    try:
        # Get the filters the user last applied.
        filters = state.get("selected_filters", {})
        
        # Format filters into a human-readable string for the prompt.
        # This handles cases where filters might be empty.
        if filters:
            filter_str = ", ".join([f"{key.replace('_', ' ')}: {value}" for key, value in filters.items()])
        else:
            filter_str = "No specific filters applied."

        conversation_history = format_conversation_with_filters(
            state["messages"],
            filters
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", NO_RESULTS_HANDLER_SYSTEM_PROMPT),
            # Use a generic "human" message, as the real context is in the system prompt
            ("human", "Please generate a helpful response for the user.")
        ])

        no_results_chain = prompt | llm | StrOutputParser()

        # Invoke the chain with all the necessary context
        response = no_results_chain.invoke({
            "inventory_summary": INVENTORY_SUMMARY,
            "filters": filter_str,
            "conversation_history": conversation_history
        })

        return {"final_answer": response}

    except Exception as e:
        logger.error(f"No results handler error: {e}")
        # An improved, more helpful fallback message
        return {"final_answer": (
            "I'm sorry, I couldn't find any vehicles that match your current criteria. "
            "Our inventory is strongest in Ford and Kia SUVs and Crossovers from the 2025 and 2026 model years."
            "Would you be interested in exploring some of our most popular models in those categories?"
        )}