````markdown
# 🚗 AI Car Buying Assistant

A Streamlit‑based AI agent that helps users find the perfect car from a dealership’s inventory. It leverages Google’s Gemini model and the LangGraph framework to deliver a conversational experience—from initial query to detailed vehicle recommendations.

---

## 🔍 Features

- **Conversational AI**  
  Chat with an expert‑level assistant trained on automotive inventory.

- **Synchronized Filters**  
  Sidebar filters (price, make, condition, etc.) that the AI understands and applies.

- **Database Integration**  
  The AI generates and executes SQL queries against a local SQLite database of vehicle inventory.

- **Detailed Summaries**  
  Rich, formatted summaries of top matches—pricing, offers, clickable links, and more.

- **Smart Error Handling**  
  If no vehicles match the criteria, the AI suggests how to broaden or refine your search.

---

## 🛠️ Setup & Installation

Follow these steps to get the project running locally.

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd ai-car-assistant
````

2. **Create and activate a virtual environment**

   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the vehicle inventory database**

   ```bash
   python setup_database.py
   ```
    You should see a confirmation that `car_inventory.db` was created successfully.

5. **Configure environment variables**

    Add your GOOGLE_API_KEY in nodes.py file
      ```ini
      GOOGLE_API_KEY="AIzaSy...your...key"
      ```

6. **Start the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   Open your browser to `http://localhost:8501` to interact with the assistant.

---

## ⚙️ How It Works

The AI agent is orchestrated as a LangGraph state machine:

1. **Planner**
   Analyzes user input and selected filters to decide whether to ask a clarifying question or proceed to search.

2. **SQL Generator**
   Crafts a precise SQLite query based on conversation context and filter values.

3. **SQL Executor**
   Runs the generated query against `car_inventory.db`.

4. **Decider**
   Checks whether results were returned:

   * **Summarizer (Results Found):** Builds a rich summary of the best-matching vehicles.
   * **No Results Handler (No Matches):** Offers friendly suggestions for adjusting the search.

---

## 📈 Conversation Workflow

![Conversation Workflow](workflow_graph.png)

---

*Happy car hunting!*