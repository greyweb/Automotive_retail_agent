import streamlit as st
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Import agent components from other files
from nodes import (
    AgentState,
    planner_node,
    sql_generator_node,
    sql_executor_node,
    summarizer_node,
    no_results_handler_node
)

# --- Logger setup ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logger = logging.getLogger("AICarAgent")

# ================================
# 1. DEBUGGING & WORKFLOW SETUP
# ================================

def debug_state(state, step, debug=True):
    if debug:
        # print("#"*50)
        # print(f"STATE AFTER {step}")
        # print(state)
        # print("#"*50)
        try:
            # Create a serializable version of the state for st.json
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
            st.write(f"Debug state display error: {e}")

    logger.info(f"STATE VALUES : {state}")
    logger.info(f"STATE AFTER {step}: Keys={list(state.keys())}")


def wrap_with_debug(node_func, step_name, debug=False):
    def wrapped_node(state):
        result = node_func(state)
        debug_state({**state, **result}, step_name, debug)
        return result
    return wrapped_node

def should_generate_sql(state: AgentState):
    """Enhanced decision logic."""
    try:
        last_message = state["messages"][-1].content
        if "GENERATE_SQL" in last_message:
            return "continue"
        return "end"
    except (IndexError, AttributeError) as e:
        logger.error(f"Decision logic error (should_generate_sql): {e}")
        return "end"

def decide_after_sql(state: AgentState):
    """Decides whether to summarize results or handle a no-results scenario."""
    query_result = state.get("query_result", "")
    if "No vehicles found" in query_result or not query_result.strip():
        logger.info("Decision: No results found. Routing to no_results_handler.")
        return "handle_no_results"
    else:
        logger.info("Decision: Results found. Routing to summarizer.")
        return "summarize"

@st.cache_resource
def create_workflow(debug=False):
    """Create and compile the workflow graph with a no-results branch."""
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", wrap_with_debug(planner_node, "PLANNER", debug))
    workflow.add_node("sql_generator", wrap_with_debug(sql_generator_node, "SQL_GENERATOR", debug))
    workflow.add_node("sql_executor", wrap_with_debug(sql_executor_node, "SQL_EXECUTOR", debug))
    workflow.add_node("summarizer", wrap_with_debug(summarizer_node, "SUMMARIZER", debug))
    workflow.add_node("no_results_handler", wrap_with_debug(no_results_handler_node, "NO_RESULTS_HANDLER", debug))
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", should_generate_sql, {"continue": "sql_generator", "end": END})
    workflow.add_edge("sql_generator", "sql_executor")
    workflow.add_conditional_edges("sql_executor", decide_after_sql, {"summarize": "summarizer", "handle_no_results": "no_results_handler"})
    workflow.add_edge("summarizer", END)
    workflow.add_edge("no_results_handler", END)
    return workflow.compile()

# ================================
# 2. STREAMLIT UI COMPONENTS
# ================================

def show_buying_guide():
    """Display car buying tips specific to our inventory."""
    st.subheader("🎓 Concord Dealership Car Buying Guide")
    
    with st.expander("🔍 What to Inspect at Our Dealership"):
        st.markdown("""
        **New Vehicles (2542 available):**
        - Verify all factory options and trim features.
        - Check for transport damage or lot wear.
        
        **Used Vehicles (292 available):**
        - Request Carfax/AutoCheck vehicle history report.
        - Inspect tires for even wear patterns.
        
        **Certified Pre-Owned (143 available):**
        - Verify CPO warranty coverage and terms.
        - Review the multi-point inspection checklist.
        """)
    
    with st.expander("💡 Negotiation Tips for Our Inventory"):
        st.markdown("""
        - **Research Our Inventory:** Use our filters to compare similar vehicles and check net_price vs MSRP.
        - **Focus on Total Price:** Negotiate the total vehicle price, not just the monthly payment.
        - **Financing:** Get pre-approved for a loan to know your budget, but also ask about our available financing options.
        """)
    
    with st.expander("⚡ Electric & Hybrid Guide (623 available)"):
        st.markdown("""
        - **Charging:** Consider home charging setup requirements and costs.
        - **Range:** Review the vehicle's range and local charging network access.
        - **Incentives:** Ask about federal and state tax incentives that may apply.
        """)

# ================================
# 3. MAIN APPLICATION LOGIC
# ================================

def main():
    st.set_page_config(
        page_title="🚗 AI Car Buying Assistant",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🚗 AI Car Buying Assistant")
    st.markdown("### Find your perfect vehicle from our Concord dealership inventory")
    
    # --- Session State Initialization ---
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {
            "messages": [], "sql_query": "", "query_result": "", "final_answer": "",
            "user_preferences": {}, "conversation_stage": "greeting", "selected_filters": {}
        }
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'workflow_app' not in st.session_state:
        st.session_state.workflow_app = create_workflow(debug=False)
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False

    # --- Sidebar for Filters ---
    with st.sidebar:
        st.header("🔍 Vehicle Filters")
        price_range = st.select_slider("Price Range", options=["Under $20k", "$20k-$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "Over $100k"], value="$30k-$50k")
        condition = st.radio("Vehicle Condition", ["Any", "New", "Used", "Certified Pre-Owned"], index=0)
        body_style = st.multiselect("Body Style", ["SUV/Crossover", "Sedan", "Truck", "Coupe", "Hatchback", "Convertible"], default=[])
        make = st.multiselect("Preferred Brands", ["Chevrolet", "Ford", "Hyundai", "Kia", "Cadillac", "BMW", "Mercedes", "Audi", "Toyota", "Honda", "Nissan"], default=[])
        fuel_type = st.multiselect("Fuel Preference", ["Gasoline Fuel", "Electric", "Hybrid"], default=[])
        
        st.session_state.conversation_state["selected_filters"] = {
            "price_range": price_range,
            "condition": condition if condition != "Any" else None,
            "body_style": body_style, "make": make, "fuel_type": fuel_type,
            "location": "Concord"
        }

    # --- Main Area ---
    if st.sidebar.button("🔄 Restart Conversation", use_container_width=True):
        st.session_state.conversation_state = {
            "messages": [], "sql_query": "", "query_result": "", "final_answer": "",
            "user_preferences": {}, "conversation_stage": "greeting",
            "selected_filters": st.session_state.conversation_state["selected_filters"]
        }
        st.session_state.chat_history = []
        st.session_state.conversation_ended = False
        st.rerun()

    tab_chat, tab_insights, tab_guide = st.tabs(["💬 Chat Advisor", "📊 Inventory Insights", "🎓 Buying Guide"])

    with tab_chat:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        chat_placeholder = "Conversation has ended. Please restart to begin." if st.session_state.conversation_ended else "Ask about vehicles or your needs..."
        user_input = st.chat_input(chat_placeholder, disabled=st.session_state.conversation_ended)
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_state["messages"].append(HumanMessage(content=user_input))
            
            with st.spinner("Your AI advisor is thinking..."):
                try:
                    result = st.session_state.workflow_app.invoke(dict(st.session_state.conversation_state))
                    
                    if result:
                        st.session_state.conversation_state.update(result)
                    
                    if result.get("final_answer"):
                        assistant_response = result["final_answer"]
                        st.session_state.conversation_ended = True
                    elif result.get("messages"):
                        assistant_response = result["messages"][-1].content
                    else: # Handle cases where graph ends without a message
                        assistant_response = "I'm ready when you are. What are you looking for?"
                        st.session_state.conversation_ended = True

                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
                    logger.error(f"Main processing error: {e}", exc_info=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": "I apologize, but I encountered an error. Please try restarting."})
            
            st.rerun()

    with tab_insights:
        st.subheader("Our Concord Inventory at a Glance")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Vehicles", "~3,000")
        metric_cols[1].metric("New Vehicles", "2,542", "85% of inventory")
        metric_cols[2].metric("SUVs/Crossovers", "1,501", "Most popular")
        metric_cols[3].metric("Eco-Friendly", "623", "Electric & Hybrid")
        st.info("🏆 **Top Brands:** Chevrolet (750), Ford (457), Hyundai (456)")

    with tab_guide:
        show_buying_guide()


if __name__ == "__main__":
    main()