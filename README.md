# 🚗 AI Car Buying Assistant

This is a Streamlit-based AI agent designed to help users find the perfect car from a dealership's inventory. The agent uses Google's Gemini model and the LangGraph framework to create a conversational experience, guiding users from their initial query to detailed vehicle recommendations.

## Features

-   **Conversational AI:** Chat with an expert AI assistant.
-   **Synchronized Filters:** Use sidebar filters (price, condition, make, etc.) that the AI understands and incorporates into its search.
-   **Database Integration:** The AI generates and executes SQL queries against a local SQLite database of vehicle inventory.
-   **Detailed Summaries:** Receive rich, formatted summaries of top vehicle matches, including pricing, offers, and clickable links.
-   **Smart Error Handling:** If no vehicles match the criteria, the AI provides helpful suggestions to broaden the search.
-   **Secure:** Uses environment variables to protect API keys.

## 🛠️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ai-car-assistant
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

You need a Google API key for the Gemini model.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Open the new `.env` file in a text editor.
3.  Replace `YOUR_GOOGLE_API_KEY_HERE` with your actual Google API key.
    ```
    GOOGLE_API_KEY="AIzaSy...your...key"
    ```
    **Note:** The `.gitignore` file is configured to ignore `.env`, so your key will not be committed to Git.

### 5. Create the Vehicle Database

Run the provided script to create and populate the `car_inventory.db` SQLite database file.

```bash
python setup_database.py
```

You should see a confirmation message that the database was created successfully.

### 6. Run the Application

You are now ready to start the Streamlit application.

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## ⚙️ How It Works

The application uses a `LangGraph` state machine to manage the conversation flow:

1.  **Planner:** Analyzes the user's input and filter selections to decide if it has enough information to search or if it needs to ask a clarifying question.
2.  **SQL Generator:** If ready to search, it constructs a precise SQLite query based on the conversation and filters.
3.  **SQL Executor:** Runs the query against the `car_inventory.db`.
4.  **Decider:** Checks if the query returned results.
    -   **Summarizer (Results Found):** Creates a rich, detailed summary of the best matching vehicles.
    -   **No Results Handler (No Results):** Generates a helpful, empathetic response with suggestions on how to modify the search.