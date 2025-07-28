# Enhanced AI Car Buying Assistant with Streamlit Interface - UPDATED VERSION
# Features: Synchronized filters, updated prompts, enhanced automotive expertise, no-results handler

import streamlit as st
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, TypedDict, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import re
import json
from datetime import datetime
from PIL import Image
import io
from urllib.parse import urljoin, urlparse
import logging

# --- Logger setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logger = logging.getLogger("AICarAgent")

def debug_state(state, step, debug=True):
    if debug:
        print("#"*50)
        print(step)
        print("#"*50)
        st.write(f"### Debug: {step}")
        try:
            serializable_state = {}
            for key, value in state.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_state[key] = value
                elif isinstance(value, pd.DataFrame):
                    serializable_state[key] = f"DataFrame with {len(value)} rows"
                else:
                    serializable_state[key] = str(type(value))
            st.json(serializable_state)
        except Exception as e:
            st.write(f"Debug state error: {e}")
    
    try:
        logger.info(f"STATE AFTER {step}: Keys={list(state.keys())}")
    except Exception as e:
        logger.error(f"Logging error in {step}: {e}")

def wrap_with_debug(node_func, step_name, debug=True):
    def wrapped_node(state):
        result = node_func(state)
        debug_state({**state, **result}, step_name, debug)
        return result
    return wrapped_node

# ================================
# 1. CONFIGURATION AND SETUP
# ================================

# Page configuration
st.set_page_config(
    page_title="🚗 AI Car Buying Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
GOOGLE_API_KEY = 'AIzaSyDkeo5SHfI_hpqgH2TlRbbvF4VwIQFUi7k'


# UPDATED: Enhanced Vehicle Database Schema synchronized with actual column descriptions
VEHICLE_SCHEMA = """
Table: car_details
Columns:
- location (TEXT): Dealership location, e.g., Concord (2977 records)
- year (INTEGER): Model year, ranging from 2022 to 2026 (Median: 2025)
- make (TEXT): Manufacturer, e.g., Chevrolet (750), Ford (457), Hyundai (456), Kia (328), Cadillac (309), etc.
- model (TEXT): Model name, e.g., Silverado 3500 HD, Silverado 2500 HD
- trim (TEXT): Trim level, e.g., WT, LT
- vin (TEXT): Vehicle Identification Number, e.g., 1GC5ARE76SF199892, 2GC4KNEY7S1148256
- url (TEXT): Vehicle details page URL
- condition (TEXT): Vehicle condition, e.g., New (2542), Used (292), Certified Pre-Owned (143)
- body_style (TEXT): Body style, e.g., SUV (773), Crossover (728), Sport Utility (570), Sedan (269)
- ext_color (TEXT): Exterior color
- int_color (TEXT): Interior color and material
- engine (TEXT): Engine specifications
- transmission (TEXT): Transmission and drivetrain
- fuel_type (TEXT): Fuel type, e.g., Gasoline Fuel (2242), Electric (322), Hybrid (301)
- msrp (REAL): Manufacturer's Suggested Retail Price
- net_price (REAL): Actual selling price
- availability (TEXT): Stock status
- offers (TEXT): Available promotions and incentives
- finance_options (TEXT): Available financing terms

IMPORTANT DATA INSIGHTS:
- Total inventory: ~3,000 vehicles across all locations
- Primary location: Concord with majority of inventory
- Year range: 2022-2026 (mostly 2025 models)
- Top brands: Chevrolet, Ford, Hyundai, Kia, Cadillac
- Condition distribution: Mostly New (2542), some Used (292), fewer CPO (143)
- Body styles: SUV/Crossover most popular, followed by Sport Utility and Sedan
- Fuel types: Predominantly Gasoline, growing Electric and Hybrid segments
"""

# ================================
# 2. DATABASE AND UTILITY FUNCTIONS
# ================================

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

# ================================
# 3. ENHANCED AGENT STATE AND LLM
# ================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    sql_query: str
    query_result: str
    final_answer: str
    user_preferences: Dict[str, Any]
    conversation_stage: str
    selected_filters: Dict[str, Any]

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

llm = get_llm()

# ================================
# 4. UPDATED WORKFLOW NODES WITH SYNCHRONIZED PROMPTS
# ================================

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

def planner_node(state: AgentState):
    """UPDATED: Enhanced planner synchronized with actual database structure."""
    try:
        conversation_history = format_conversation_with_filters(
            state["messages"], 
            state.get("selected_filters", {})
        )
        
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert automotive sales consultant with 15+ years of experience specializing in our inventory of ~3,000 vehicles primarily located in Concord.

INVENTORY EXPERTISE:
- Location: Primarily Concord dealership with comprehensive inventory
- Model Years: 2022-2026 (majority are 2025 models)
- Top Brands: Chevrolet (750), Ford (457), Hyundai (456), Kia (328), Cadillac (309)
- Condition Mix: Mostly New (2542), Used (292), Certified Pre-Owned (143)
- Body Styles: SUV (773), Crossover (728), Sport Utility (570), Sedan (269)
- Fuel Types: Gasoline Fuel (2242), Electric (322), Hybrid (301)

CONSULTATION STRATEGY:
1. Acknowledge any customer filter selections immediately
2. Understand complete needs: budget (net_price), condition preference, body_style, fuel_type
3. Guide toward vehicles matching lifestyle and budget from our actual inventory
4. Emphasize value propositions and financing options available

REQUIRED INFO FOR RECOMMENDATIONS:
- Budget range (net_price preference)
- Condition: New/Used/Certified Pre-Owned
- Body style OR specific use case requirements
- Fuel type preference (especially for electric/hybrid interest)
- Location preference (primarily Concord available)

DATABASE SCHEMA:
{schema}

CURRENT CONVERSATION WITH FILTERS:
{conversation_history}

DECISION LOGIC:
- If you have sufficient information (budget + condition + body style OR use case), respond with "GENERATE_SQL"
- Otherwise, ask ONE focused question about missing critical information
- Be consultative about our specific inventory strengths
- Focus on features, reliability, financing options, and trim levels available

Strategic Questions Examples:
- "I see you're interested in SUVs. We have 773 SUVs and 728 Crossovers in stock. What's your budget range?"
- "For your budget, would you prefer a new 2025 model with full warranty, or a used vehicle for better value?"
- "Are you interested in our electric (322 available) or hybrid (301 available) options for fuel efficiency?"
"""),
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
            ("system", """You are an expert SQL developer specializing in automotive inventory systems with deep knowledge of our specific database structure.

DATABASE SCHEMA WITH ACTUAL DATA PATTERNS:
{schema}

FILTER INTEGRATION RULES:
1. ALWAYS use customer filter selections as primary constraints
2. Map filter values to exact database values
3. Use conversation context for secondary refinement

COLUMN MAPPING AND QUERY LOGIC:

1. PRICE FILTERING (use net_price column):
   - "Under $20k": net_price < 20000
   - "$20k-$30k": net_price BETWEEN 20000 AND 30000
   - "$30k-$50k": net_price BETWEEN 30000 AND 50000
   - "$50k-$75k": net_price BETWEEN 50000 AND 75000
   - "$75k-$100k": net_price BETWEEN 75000 AND 100000
   - "Over $100k": net_price > 100000

2. CONDITION FILTERING (exact values):
   - Use exact values: 'New', 'Used', 'Certified Pre-Owned'

3. BODY_STYLE MAPPING (use actual database values):
   - "SUV/Crossover" → body_style IN ('SUV', 'Crossover', 'Sport Utility')
   - "Sedan" → body_style = 'Sedan'
   - "Truck" → body_style LIKE '%Truck%' OR body_style LIKE '%Cab%'
   - Map other styles to exact database values

4. MAKE FILTERING:
   - Use exact manufacturer names from database
   - Top makes: 'Chevrolet', 'Ford', 'Hyundai', 'Kia', 'Cadillac'

5. FUEL_TYPE FILTERING:
   - Use exact values: 'Gasoline Fuel', 'Electric', 'Hybrid'

6. LOCATION:
   - Primary location is 'Concord' (2977 records)

OPTIMIZATION RULES:
- ORDER BY best value within budget criteria
- Consider both net_price and available offers
- LIMIT 5
- Include key columns: year, make, model, trim, net_price, msrp, condition, body_style, fuel_type, offers, finance_options, url

CONVERSATION AND FILTER CONTEXT:
{conversation_history}

Generate precise SQLite query incorporating ALL filters and conversation context.
Use exact column names and values from the schema.
Output ONLY the SQL query, no explanations.
"""),
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
        
        # Ensure all necessary columns are present before creating markdown table
        required_cols = ['make', 'model', 'year', 'net_price', 'url']
        for col in required_cols:
            if col not in result_df.columns:
                result_df[col] = 'N/A' # Add missing columns with a placeholder

        markdown_table = result_df.to_markdown(index=False)
        return {"query_result": markdown_table}
        
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return {"query_result": f"Error executing query: {str(e)}"}

def summarizer_node(state: AgentState):
    """UPDATED: Creates a highly detailed summary with clickable links, financial details, and pros/cons for 2 cars, plus a table for 5."""
    try:
        conversation_history = format_conversation_with_filters(
            state["messages"],
            state.get("selected_filters", {})
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert automotive sales consultant. Your goal is to create a rich, detailed, and helpful summary of vehicle search results for the user.

**SEARCH RESULTS (in markdown format):**
{query_result}

**CONVERSATION HISTORY:**
{conversation_history}

**YOUR TASK - Follow this structure precisely:**

**Part 1: Detailed Recommendations (Top 2 Cars)**
- From the search results, select the **two most relevant cars** based on the user's request.
- For each car, create a detailed profile using the exact format below.

**Part 2: Summary Table (Top 5 Cars)**
- After the detailed profiles, provide a markdown table of the **top 5 vehicles** from the search results.

---
**DETAILED PROFILE FORMAT (Use for each of the top 2 cars):**

### [Year Make Model Trim](url)
*   **Key Details:**
    *   **Pricing:** **$[net_price]** (MSRP: $[msrp] - *You Save $[msrp - net_price]*)
    *   **Condition & Fuel:** [Condition], [Fuel Type]
    *   **Offers:** [List any offers from the 'offers' column, or state 'None listed'.]
    *   **Financing:** [List any terms from the 'finance_options' column, or state 'See dealer for options'.]
*   **Highlights:** [Provide a 1-2 sentence summary of why this car is a great choice. Mention value, features, warranty, or a great deal.]
*   **Considerations:** [Provide a 1-sentence note on any potential trade-offs, like a higher price, basic trim, etc.]

---
**SUMMARY TABLE FORMAT (Use for the top 5 cars):**

| Make | Model | Year | Net Price | Link |
|---|---|---|---|---|
| [Make] | [Model] | [Year] | $[Net Price] | [View Details](url) |

---
**RESPONSE STRUCTURE EXAMPLE:**

Based on your search for a family SUV, here are my top recommendations from our inventory:

### [2025 Chevrolet Equinox LT](https://www.example.com/link-to-car-1)
*   **Key Details:**
    *   **Pricing:** **$35,123** (MSRP: $37,000 - *You Save $1,877*)
    *   **Condition & Fuel:** New, Gasoline Fuel
    *   **Offers:** $1,500 Customer Cash available.
    *   **Financing:** 1.9% APR for 60 months for qualified buyers.
*   **Highlights:** This brand-new SUV offers tremendous value with significant savings and excellent financing. It's a perfect choice for families seeking reliability and a full factory warranty.
*   **Considerations:** As an LT trim, it may not have all the luxury features found in the Premier models.

### [2025 Hyundai Santa Fe SEL](https://www.example.com/link-to-car-2)
*   **Key Details:**
    *   **Pricing:** **$38,456** (MSRP: $39,500 - *You Save $1,044*)
    *   **Condition & Fuel:** New, Hybrid
    *   **Offers:** None listed.
    *   **Financing:** See dealer for financing options.
*   **Highlights:** This modern hybrid SUV is packed with the latest technology and offers fantastic long-term fuel savings. The SEL trim provides a great balance of features for the price.
*   **Considerations:** While the initial savings are lower, the hybrid engine provides excellent cost-of-ownership benefits.

---
**Top 5 Matches from Our Inventory:**

| Make      | Model      | Year | Net Price | Link          |
|-----------|------------|------|-----------|---------------|
| Chevrolet | Equinox    | 2025 | $35,123   | [View Details](https://www.example.com/link-to-car-1) |
| Hyundai   | Santa Fe   | 2025 | $38,456   | [View Details](https://www.example.com/link-to-car-2) |
| Ford      | Escape     | 2025 | $34,999   | [View Details](https://www.example.com/link-to-car-3) |
| Kia       | Sorento    | 2025 | $37,888   | [View Details](https://www.example.com/link-to-car-4) |
| Cadillac  | XT4        | 2025 | $42,111   | [View Details](https://www.example.com/link-to-car-5) |

"""),
            ("human", "Please create the detailed summary as instructed using the provided search results and conversation context.")
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
            "Sorry, I couldn't prepare your detailed recommendations at this time. Please try again."
        )}

# ================================
# 5. NEW: NODE FOR HANDLING NO RESULTS
# ================================

def no_results_handler_node(state: AgentState):
    """
    Handles the scenario where the SQL query returns no results.
    Provides helpful, strategic suggestions to the user.
    """
    try:
        conversation_history = format_conversation_with_filters(
            state["messages"],
            state.get("selected_filters", {})
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert, empathetic automotive sales consultant. A search of our inventory based on the user's request has returned zero results.

Your task is to craft a helpful and strategic response. DO NOT sound like a robot saying "no results found."

**Analysis of the situation:**
- The user's filters or request were too specific for our current inventory.
- We need to guide them toward alternatives without losing their interest.

**RESPONSE STRATEGY:**
1.  **Acknowledge and Empathize:** Start by politely informing the user that you couldn't find an *exact* match for their specific criteria.
2.  **Identify the Constraints:** Briefly mention the combination of filters that might be too restrictive (e.g., "a new electric truck under $50,000").
3.  **Offer Smart, Actionable Alternatives:**
    *   Suggest loosening the *least important* filter. For example, if they asked for a specific color, suggest looking at other colors. If they set a very tight budget, suggest expanding it slightly.
    *   Propose a related alternative that IS in our inventory. Use your knowledge of our top brands (Chevrolet, Ford, Hyundai, Kia) and popular styles (SUVs, Crossovers).
    *   Ask a question to re-engage the user and guide them toward these alternatives.

**INVENTORY KNOWLEDGE:**
- We have ~3,000 vehicles, mostly 2025 models.
- Top Brands: Chevrolet, Ford, Hyundai, Kia, Cadillac.
- Popular Styles: SUV, Crossover, Sport Utility.
- Condition: Mostly New.
- Fuel Types: Strong selection of Gasoline, growing Electric & Hybrid.

**USER's REQUEST (CONVERSATION & FILTERS):**
{conversation_history}

**EXAMPLE RESPONSES:**

*   **Scenario (Too specific):** User wants a "New 2025 Red Ford F-150 under $40,000".
*   **Good Response:** "I couldn't find a new 2025 red Ford F-150 in our inventory under $40,000 at the moment. We do have several new Chevrolet Silverado and Ford Ranger models in that price range that are very popular. Would you be interested in exploring some of those options, or perhaps looking at other colors for the F-150?"

*   **Scenario (Niche request):** User wants a "Used convertible under $25,000".
*   **Good Response:** "It looks like we don't have any used convertibles that match your budget right now. However, for under $25,000, we have an excellent selection of sporty new sedans and used coupes from brands like Hyundai and Kia that offer a fun driving experience. Would you be open to considering a stylish coupe instead?"

---
Now, craft a response for the current user's request. Be friendly, helpful, and strategic.
"""),
            ("human", "Generate a helpful response for the user since no cars were found.")
        ])

        no_results_chain = prompt | llm | StrOutputParser()
        response = no_results_chain.invoke({
            "conversation_history": conversation_history
        })

        return {"final_answer": response}

    except Exception as e:
        logger.error(f"No results handler error: {e}")
        return {"final_answer": (
            "I couldn't find any vehicles that exactly match your criteria. "
            "You might want to try adjusting your filters, such as widening the price range or selecting a different body style."
        )}

# ================================
# 6. UPDATED: WORKFLOW SETUP
# ================================

def should_generate_sql(state: AgentState):
    """Enhanced decision logic."""
    try:
        last_message = state["messages"][-1].content
        return "continue" if "GENERATE_SQL" in last_message else "end"
    except Exception as e:
        logger.error(f"Decision logic error: {e}")
        return "end"

# --- NEW: DECISION FUNCTION AFTER SQL EXECUTION ---
def decide_after_sql(state: AgentState):
    """
    Decides whether to summarize results or handle a no-results scenario.
    """
    query_result = state.get("query_result", "")
    if "No vehicles found" in query_result or not query_result.strip():
        logger.info("Decision: No results found. Routing to no_results_handler.")
        return "handle_no_results"
    else:
        logger.info("Decision: Results found. Routing to summarizer.")
        return "summarize"

# --- UPDATED: WORKFLOW CREATION ---
@st.cache_resource
def create_workflow(debug=True):
    """Create and compile the workflow graph with a no-results branch."""
    workflow = StateGraph(AgentState)

    # Add all nodes, including the new one
    workflow.add_node("planner", wrap_with_debug(planner_node, "PLANNER", debug))
    workflow.add_node("sql_generator", wrap_with_debug(sql_generator_node, "SQL_GENERATOR", debug))
    workflow.add_node("sql_executor", wrap_with_debug(sql_executor_node, "SQL_EXECUTOR", debug))
    workflow.add_node("summarizer", wrap_with_debug(summarizer_node, "SUMMARIZER", debug))
    workflow.add_node("no_results_handler", wrap_with_debug(no_results_handler_node, "NO_RESULTS_HANDLER", debug)) # New node

    # Define the workflow structure
    workflow.set_entry_point("planner")

    # Conditional edge from planner to either generate SQL or end
    workflow.add_conditional_edges(
        "planner",
        should_generate_sql,
        {"continue": "sql_generator", "end": END}
    )

    # Path for generating and executing SQL
    workflow.add_edge("sql_generator", "sql_executor")

    # NEW: Conditional edge after executing SQL to handle results or no results
    workflow.add_conditional_edges(
        "sql_executor",
        decide_after_sql,
        {
            "summarize": "summarizer",
            "handle_no_results": "no_results_handler"
        }
    )

    # Endpoints for the two final branches
    workflow.add_edge("summarizer", END)
    workflow.add_edge("no_results_handler", END)

    return workflow.compile()


# ================================
# 7. UPDATED STREAMLIT APPLICATION WITH SYNCHRONIZED FILTERS
# ================================

def main():
    st.title("🚗 AI Car Buying Assistant")
    st.markdown("### Find your perfect vehicle from our Concord dealership inventory")
    st.markdown("*Over 2,900 vehicles available with expert financing guidance*")
    
    # Initialize session state
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {
            "messages": [],
            "sql_query": "",
            "query_result": "",
            "final_answer": "",
            "user_preferences": {},
            "conversation_stage": "greeting",
            "selected_filters": {}
        }
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'workflow_app' not in st.session_state:
        st.session_state.workflow_app = create_workflow(debug=False) # Set debug to False for cleaner UI
    
    # UPDATED: Sidebar with synchronized filters based on actual data
    with st.sidebar:
        st.header("🔍 Vehicle Filters")
        st.markdown("*Filters based on actual inventory*")
        
        # Price Range - aligned with SQL mapping
        st.subheader("💰 Budget")
        price_range = st.select_slider(
            "Price Range",
            options=["Under $20k", "$20k-$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "Over $100k"],
            value="$30k-$50k"
        )
        
        # Condition - exact database values
        st.subheader("🆕 Condition")
        condition = st.radio(
            "Vehicle Condition",
            ["Any", "New", "Used", "Certified Pre-Owned"],
            index=0,
            help="New: 2542 available | Used: 292 available | CPO: 143 available"
        )
        
        # Body Style - mapped to actual database values
        st.subheader("🚙 Body Style")
        body_style = st.multiselect(
            "Body Style",
            ["SUV/Crossover", "Sedan", "Truck", "Coupe", "Hatchback", "Convertible"],
            default=[],
            help="SUV: 773 | Crossover: 728 | Sport Utility: 570 | Sedan: 269"
        )
        
        # Make/Brand - top brands from actual data
        st.subheader("🏭 Brand")
        make = st.multiselect(
            "Preferred Brands",
            ["Chevrolet", "Ford", "Hyundai", "Kia", "Cadillac", "BMW", "Mercedes", "Audi", "Toyota", "Honda", "Nissan"],
            default=[],
            help="Top inventory: Chevrolet (750), Ford (457), Hyundai (456), Kia (328), Cadillac (309)"
        )
        
        # Fuel Type - actual database values
        st.subheader("⛽ Fuel Type")
        fuel_type = st.multiselect(
            "Fuel Preference",
            ["Gasoline Fuel", "Electric", "Hybrid"],
            default=[],
            help="Gasoline: 2242 | Electric: 322 | Hybrid: 301"
        )
        
        # Location - primarily Concord
        st.subheader("📍 Location")
        location = st.selectbox(
            "Dealership Location",
            ["Concord", "Any Location"],
            index=0,
            help="Primary inventory at Concord location (2977 vehicles)"
        )
        
        # Update filters in session state
        st.session_state.conversation_state["selected_filters"] = {
            "price_range": price_range,
            "condition": condition if condition != "Any" else None,
            "body_style": body_style,
            "make": make,
            "fuel_type": fuel_type,
            "location": location if location != "Any Location" else None,
        }
        
        # Inventory summary
        st.markdown("---")
        st.subheader("📊 Inventory Overview")
        st.info("🚗 Total: ~3,000 vehicles")
        st.info("📅 Years: 2022-2026 (mostly 2025)")
        st.info("🆕 Condition: 85% New, 10% Used, 5% CPO")
        st.info("🚙 Popular: SUVs & Crossovers")
        
        # Reset button
        if st.button("🔄 Reset Conversation"):
            st.session_state.conversation_state = {
                "messages": [],
                "sql_query": "",
                "query_result": "",
                "final_answer": "",
                "user_preferences": {},
                "conversation_stage": "greeting",
                "selected_filters": st.session_state.conversation_state["selected_filters"]
            }
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface (no columns)
    st.subheader("💬 Chat with Your Car Advisor")
    
    # MODIFICATION: Simplified chat history display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"]) # Use markdown to render clickable links
    
    # Chat input
    user_input = st.chat_input("Ask about vehicles, financing, or your car needs...")
    
    # MODIFICATION: Updated chat processing logic
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process message through workflow
        st.session_state.conversation_state["messages"].append(HumanMessage(content=user_input))
        
        with st.spinner("Searching our inventory..."):
            try:
                current_state = dict(st.session_state.conversation_state)
                
                result = st.session_state.workflow_app.invoke(current_state)
                
                if result:
                    st.session_state.conversation_state.update(result)
                
                # Get the final answer from the workflow
                if result.get("final_answer"):
                    assistant_response = result["final_answer"]
                # Handle cases where the conversation continues (planner asks a question)
                elif result.get("messages"):
                    assistant_response = result["messages"][-1].content
                else:
                    assistant_response = "I'm ready to help. What are you looking for?"

                # Update langgraph state for the next turn
                if st.session_state.conversation_state.get("messages"):
                    st.session_state.conversation_state["messages"][-1] = AIMessage(content=assistant_response)
                else:
                    st.session_state.conversation_state["messages"].append(AIMessage(content=assistant_response))
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": assistant_response,
                })
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                logger.error(f"Main processing error: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "I apologize, but I encountered an error. Please try rephrasing your question.",
                })
        
        st.rerun()

# ================================
# 8. UPDATED ADDITIONAL FEATURES
# ================================

def show_buying_guide():
    """UPDATED: Display car buying tips specific to our inventory."""
    st.subheader("🎓 Concord Dealership Car Buying Guide")
    
    with st.expander("🔍 What to Inspect at Our Dealership"):
        st.markdown("""
        **New Vehicles (2542 available):**
        - Verify all factory options and trim features
        - Check for transport damage or lot wear
        - Ensure all paperwork and warranties are complete
        - Ask about manufacturer incentives and rebates
        
        **Used Vehicles (292 available):**
        - Request Carfax/AutoCheck vehicle history report
        - Inspect tires for even wear patterns
        - Review maintenance records and service history
        - Test all electrical systems and features
        - Look for signs of accidents, flooding, or major repairs
        
        **Certified Pre-Owned (143 available):**
        - Verify CPO warranty coverage and terms
        - Review the multi-point inspection checklist
        - Understand additional benefits and roadside assistance
        """)
    
    with st.expander("💡 Negotiation Tips for Our Inventory"):
        st.markdown("""
        **Research Our Inventory:**
        - Use our filters to compare similar vehicles
        - Check net_price vs MSRP for potential savings
        - Look for vehicles with manufacturer offers
        - Compare trim levels and features
        
        **At Our Concord Dealership:**
        - Focus on total price, not just monthly payments
        - Ask about available financing options
        - Negotiate trade-in value separately
        - Consider our extensive inventory for alternatives
        - Don't rush - we have 3000+ vehicles to choose from
        """)
    
    with st.expander("⚡ Electric & Hybrid Guide (623 available)"):
        st.markdown("""
        **Electric Vehicles (322 available):**
        - Calculate charging costs vs gasoline savings
        - Consider home charging setup requirements
        - Review range and charging network access
        - Ask about federal and state tax incentives
        
        **Hybrid Vehicles (301 available):**
        - Understand different hybrid types (mild, full, plug-in)
        - Compare fuel economy improvements
        - Consider battery warranty coverage
        - Review maintenance differences
        """)

def add_sidebar_features():
    """UPDATED: Add features specific to our inventory."""
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🎓 Concord Buying Guide"):
        show_buying_guide()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Inventory Insights**")
    st.sidebar.info("🏆 85% of our inventory is new 2025 models")
    st.sidebar.info("🚙 Largest SUV/Crossover selection in the area") 
    st.sidebar.info("⚡ 623 eco-friendly electric & hybrid options")
    st.sidebar.info("🎯 143 certified pre-owned vehicles available")

# ================================
# 9. MAIN APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .stExpander > div:first-child {
        background-color: #f0f2f6;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .vehicle-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
    }
    .st-emotion-cache-1c7y2kd { /* Targets chat message container */
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .inventory-highlight {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add sidebar features
    add_sidebar_features()
    
    # Run main application
    main()
    
    # Updated footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🚗 <span class='inventory-highlight'>Concord Dealership AI Assistant</span> | 3000+ Vehicles Available<br>
    💡 Expert guidance for New, Used & Certified Pre-Owned vehicles with comprehensive financing options<br>
    📍 Visit us at Concord or browse our complete inventory online
    </div>
    """, unsafe_allow_html=True)