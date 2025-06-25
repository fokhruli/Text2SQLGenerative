from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file

import streamlit as st
import os
import pyodbc
import google.generativeai as genai
import re

# Configure the GenAI API key using environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Default database connection parameters
DEFAULT_SERVER = "192.168.100.129"
DEFAULT_USER = "sa"
DEFAULT_PASSWORD = "dataport"
DEFAULT_DATABASE = "AIDataset"
DEFAULT_TABLE = "[dbo].[BusinessOverview]"

# Define available Gemini models
gemini_models = [
    "gemini-2.0-flash-lite-001",  # Default model
    "gemini-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro"
]

# Function to get available databases
def get_databases(conn_params):
    try:
        # Create connection string
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={conn_params['server']};"
            f"UID={conn_params['user']};"
            f"PWD={conn_params['password']};"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
        databases = [row.name for row in cursor.fetchall()]
        conn.close()
        return databases
    except pyodbc.Error as e:
        st.error(f"Error fetching databases: {str(e)}")
        return []

# Function to get tables for a selected database
def get_tables(conn_params, database):
    try:
        # Create connection string with the selected database
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={conn_params['server']};"
            f"DATABASE={database};"
            f"UID={conn_params['user']};"
            f"PWD={conn_params['password']};"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Get tables with more detailed information
        cursor.execute("""
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_SCHEMA + '.' + TABLE_NAME AS schema_table,
                QUOTENAME(TABLE_SCHEMA) + '.' + QUOTENAME(TABLE_NAME) AS quoted_name
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        
        tables_info = []
        for row in cursor.fetchall():
            # Store both the display name and the proper SQL name
            display_name = f"[{row.schema_table}]"
            sql_name = row.quoted_name
            tables_info.append({
                'display': display_name,
                'sql_name': sql_name,
                'schema': row.TABLE_SCHEMA,
                'table': row.TABLE_NAME
            })
        
        conn.close()
        return tables_info
    except pyodbc.Error as e:
        st.error(f"Error fetching tables: {str(e)}")
        return []

# Function to get table schema
def get_table_schema(conn_params, database, table_info):
    """Get the schema/structure of a specific table"""
    try:
        # Extract schema and table name from table_info
        if isinstance(table_info, dict):
            schema_name = table_info['schema']
            table_name = table_info['table']
        else:
            # Fallback for old format
            clean_table_name = table_info.replace('[', '').replace(']', '')
            schema_parts = clean_table_name.split('.')
            if len(schema_parts) == 2:
                schema_name = schema_parts[0]
                table_name = schema_parts[1]
            else:
                schema_name = 'dbo'
                table_name = clean_table_name
        
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={conn_params['server']};"
            f"DATABASE={database};"
            f"UID={conn_params['user']};"
            f"PWD={conn_params['password']};"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Get column information
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """, (schema_name, table_name))
        
        columns = cursor.fetchall()
        conn.close()
        
        # Convert to list of tuples for consistent handling
        result = []
        for col in columns:
            result.append((col.COLUMN_NAME, col.DATA_TYPE, col.IS_NULLABLE, col.COLUMN_DEFAULT))
        
        return result
    except Exception as e:
        st.error(f"Error getting table schema: {str(e)}")
        return []

# Function to validate and fix SQL queries
def validate_and_fix_sql(generated_sql, correct_table_name):
    """Validate and fix the generated SQL to use the correct table name"""
    
    # Remove any SQL formatting markers - be more aggressive
    cleaned_sql = generated_sql.strip()
    
    # Remove various markdown SQL formatting patterns
    cleaned_sql = cleaned_sql.replace('````sql', '').replace('````', '')
    cleaned_sql = cleaned_sql.replace('```sql', '').replace('```', '')
    cleaned_sql = cleaned_sql.replace('``sql', '').replace('``', '')
    cleaned_sql = cleaned_sql.replace('`sql', '').replace('`', '')
    
    # Remove "sql " prefix if it exists
    if cleaned_sql.lower().startswith('sql '):
        cleaned_sql = cleaned_sql[4:]
    
    # Clean up any remaining whitespace and newlines
    cleaned_sql = cleaned_sql.strip()
    
    # Simple and more precise approach: replace table references with the correct table name
    import re
    
    # More precise regex patterns that won't double-replace
    # Pattern 1: FROM [schema].[table] or FROM [table]
    cleaned_sql = re.sub(r'FROM\s+\[[^\]]+\]\.\[[^\]]+\]', f'FROM {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'FROM\s+\[[^\]]+\](?!\.\[)', f'FROM {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    
    # Pattern 2: FROM schema.table or FROM table (without brackets)
    cleaned_sql = re.sub(r'FROM\s+[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*', f'FROM {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'FROM\s+[a-zA-Z_][a-zA-Z0-9_]*(?!\.[a-zA-Z_])', f'FROM {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    
    # Similar patterns for JOIN clauses
    cleaned_sql = re.sub(r'JOIN\s+\[[^\]]+\]\.\[[^\]]+\]', f'JOIN {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'JOIN\s+\[[^\]]+\](?!\.\[)', f'JOIN {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'JOIN\s+[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*', f'JOIN {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    
    # UPDATE patterns
    cleaned_sql = re.sub(r'UPDATE\s+\[[^\]]+\]\.\[[^\]]+\]', f'UPDATE {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'UPDATE\s+\[[^\]]+\](?!\.\[)', f'UPDATE {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    
    # INSERT INTO patterns
    cleaned_sql = re.sub(r'INSERT\s+INTO\s+\[[^\]]+\]\.\[[^\]]+\]', f'INSERT INTO {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'INSERT\s+INTO\s+\[[^\]]+\](?!\.\[)', f'INSERT INTO {correct_table_name}', cleaned_sql, flags=re.IGNORECASE)
    
    return cleaned_sql

# Function to interact with the Gemini model and get an SQL query back
def get_gemini_response(question, prompt, selected_model):
    try:
        # Use the selected model instead of hardcoding it
        model = genai.GenerativeModel(selected_model)
        response = model.generate_content([prompt[0], question])
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "ResourceExhausted" in error_msg:
            return """ERROR: API quota exceeded. 
            
Possible solutions:
1. Wait 10-15 minutes and try again
2. Switch to 'gemini-2.0-flash-lite-001' model
3. Check your API usage at https://aistudio.google.com/
4. Consider upgrading to a paid plan for higher limits"""
        elif "401" in error_msg or "authentication" in error_msg.lower():
            return "ERROR: Authentication failed. Please check your API key in the .env file."
        else:
            return f"ERROR: {error_msg}"

# Function to run SQL queries against SQL Server database
def read_sql_query(sql, conn_params):
    # Create connection string for SQL Server
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={conn_params['server']};"
        f"DATABASE={conn_params['database']};"
        f"UID={conn_params['user']};"
        f"PWD={conn_params['password']};"
    )
    
    try:
        # Connect to SQL Server
        conn = pyodbc.connect(conn_str)
        cur = conn.cursor()
        cur.execute(sql)  # Execute the query
        rows = cur.fetchall()  # Get the results
        column_names = [column[0] for column in cur.description] if cur.description else ["No columns"]
        conn.close()
        return rows, column_names
    except pyodbc.Error as e:
        return [(f"Database Error: {str(e)}",)], ["Error"]

# Streamlit app setup
st.set_page_config(page_title="SQL Query Generator")
st.title("SQL Query Generator with Gemini AI")

# Sidebar for connection settings and model selection
st.sidebar.header("Database Connection Settings")

# Connection inputs with default values
server = st.sidebar.text_input("Server", DEFAULT_SERVER)
user = st.sidebar.text_input("Username", DEFAULT_USER)
password = st.sidebar.text_input("Password", DEFAULT_PASSWORD, type="password")

# Connect button to fetch databases
if st.sidebar.button("Connect to Server"):
    conn_params = {
        "server": server,
        "user": user,
        "password": password,
        "database": ""  # No database selected yet
    }
    
    # Store connection params in session state
    st.session_state["conn_params"] = conn_params
    
    # Fetch databases
    databases = get_databases(conn_params)
    st.session_state["databases"] = databases
    
    if databases:
        st.sidebar.success("Connection successful!")
    else:
        st.sidebar.error("Failed to connect to server or no databases found.")

# Database selection dropdown (only shows if databases are fetched)
if "databases" in st.session_state and st.session_state["databases"]:
    selected_db = st.sidebar.selectbox(
        "Select Database", 
        options=st.session_state["databases"],
        index=st.session_state["databases"].index(DEFAULT_DATABASE) if DEFAULT_DATABASE in st.session_state["databases"] else 0
    )
    
    # Update connection params with selected database
    if "conn_params" in st.session_state:
        conn_params = st.session_state["conn_params"]
        conn_params["database"] = selected_db
        
        # Fetch tables when database is selected
        if st.sidebar.button("Load Tables") or "tables" not in st.session_state:
            tables = get_tables(conn_params, selected_db)
            st.session_state["tables"] = tables
            st.session_state["selected_db"] = selected_db

# Table selection (only shows if tables are fetched)
if "tables" in st.session_state and st.session_state["tables"]:
    # Create a list of display names for the selectbox
    table_display_names = [table['display'] for table in st.session_state["tables"]]
    
    selected_table_index = st.sidebar.selectbox(
        "Select Table",
        options=range(len(table_display_names)),
        format_func=lambda x: table_display_names[x],
        index=0
    )
    
    # Get the selected table info
    selected_table_info = st.session_state["tables"][selected_table_index]
    st.session_state["selected_table_info"] = selected_table_info

# Add model selection to the sidebar
st.sidebar.header("Model Settings")
selected_model = st.sidebar.selectbox(
    "Select Gemini Model",
    options=gemini_models,
    index=0,  # Default to the first model in the list
    help="Choose which Google Gemini model to use for query generation"
)

# Main app area
st.write("Enter a question about your database, and this app will generate the corresponding SQL query.")

# Show current connection info
if "selected_db" in st.session_state and "selected_table_info" in st.session_state:
    table_info = st.session_state["selected_table_info"]
    st.info(f"Connected to: {st.session_state['selected_db']} | Table: {table_info['display']}")
    
    # Add a button to show table schema
    if st.button("Show Table Schema"):
        conn_params = st.session_state["conn_params"]
        database = st.session_state["selected_db"]
        
        schema_info = get_table_schema(conn_params, database, table_info)
        if schema_info:
            st.subheader(f"Schema for {table_info['display']}:")
            
            # Create a DataFrame to display schema nicely
            import pandas as pd
            try:
                # Check if we have the expected number of columns
                if len(schema_info) > 0:
                    num_cols = len(schema_info[0])
                    if num_cols == 4:
                        schema_df = pd.DataFrame(schema_info, columns=['Column Name', 'Data Type', 'Nullable', 'Default'])
                    elif num_cols == 3:
                        schema_df = pd.DataFrame(schema_info, columns=['Column Name', 'Data Type', 'Nullable'])
                    elif num_cols == 2:
                        schema_df = pd.DataFrame(schema_info, columns=['Column Name', 'Data Type'])
                    else:
                        # Fallback: just show the raw data
                        schema_df = pd.DataFrame(schema_info)
                    st.dataframe(schema_df)
                else:
                    st.write("No schema information available.")
            except Exception as e:
                st.error(f"Error displaying schema: {str(e)}")
                st.write("Raw schema data:")
                for i, col in enumerate(schema_info):
                    st.write(f"{i+1}. {col}")
            
            # Show the exact SQL table name that will be used
            st.info(f"SQL Table Name: {table_info['sql_name']}")
        else:
            st.error("Could not retrieve table schema")
            
else:
    st.info(f"Using default connection: {DEFAULT_DATABASE} | Table: {DEFAULT_TABLE}")

# Define some example questions
# Define some example questions
example_questions = [
    "How many records are in the table?",
    "Show me the top 10 records",
    "Show me the first 5 rows", 
    "What are the column names?",
    "Show me all the data",
    "Show me the last 10 records",
    "Show me 20 records",
    "Display a sample of data"
]
# Helper function to set the question
def set_question(question_text):
    st.session_state.user_question = question_text

# Initialize session state for the question
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

# Create a small header for the question suggestions
st.write("**Suggested questions:**")

# Create a container for the example buttons in a single row
example_container = st.container()

# Create a horizontal layout for suggestion buttons
cols = example_container.columns(4)
for i, col in enumerate(cols):
    for j in range(2):  # Two questions per column
        idx = i * 2 + j
        if idx < len(example_questions):
            col.button(
                example_questions[idx], 
                key=f"example_{idx}",
                on_click=set_question,
                args=(example_questions[idx],)
            )

# Text input that can be prefilled by suggestion buttons or typed in directly
question = st.text_input(
    "Your question:", 
    value=st.session_state.user_question,
    placeholder="Type your question or select a suggestion above"
)

# Inform the user which model is being used
st.caption(f"Using Gemini model: **{selected_model}**")

# Button to submit the question
submit = st.button("Generate SQL Query")

# When the button is pressed and the user has entered a question
if submit and question:
    # Determine which connection and table to use
    if "conn_params" in st.session_state and "selected_db" in st.session_state and "selected_table_info" in st.session_state:
        # Use selected connection
        conn_params = st.session_state["conn_params"]
        database = st.session_state["selected_db"]
        table_info = st.session_state["selected_table_info"]
        table_sql_name = table_info['sql_name']  # This is the proper SQL name
        table_display = table_info['display']
    else:
        # Use default connection
        conn_params = {
            "server": DEFAULT_SERVER,
            "user": DEFAULT_USER,
            "password": DEFAULT_PASSWORD,
            "database": DEFAULT_DATABASE
        }
        database = DEFAULT_DATABASE
        table_sql_name = DEFAULT_TABLE
        table_display = DEFAULT_TABLE
        table_info = None
    
    # Get table schema for better prompt
    schema_info = get_table_schema(conn_params, database, table_info or table_sql_name)
    schema_text = ""
    if schema_info:
        schema_text = "The table has the following columns:\n"
        for col in schema_info:
            schema_text += f"- {col[0]} ({col[1]})\n"
    
    # Define the enhanced prompt with very specific instructions
    prompt = [
        f"""
        You are a SQL generator. You must follow these rules EXACTLY:
        
        1. ONLY use this table name: {table_sql_name}
        2. Copy the table name exactly as shown: {table_sql_name}
        3. Do NOT use any other table names
        4. Do NOT modify the table name format 
        5. Return ONLY a SQL query, no explanations or formatting
        
        Database: {database}
        Required Table Name: {table_sql_name}
        
        {schema_text}
        
        Question: {question}
        
        Generate a SQL Server query using ONLY the table name {table_sql_name}:
        """
    ]
    
    # Show a spinner while the AI is generating the SQL query
    with st.spinner("Generating SQL query..."):
        # Call the function to get the SQL query with selected model
        generated_sql = get_gemini_response(question, prompt, selected_model)
        
        # Check if there was an error in the response
        if generated_sql.startswith("ERROR:"):
            st.error(generated_sql)
        else:
            # Clean the original SQL first for better display
            clean_original = generated_sql.strip()
            # Remove markdown formatting for display
            clean_original = clean_original.replace('````sql', '').replace('````', '')
            clean_original = clean_original.replace('```sql', '').replace('```', '')
            clean_original = clean_original.replace('``sql', '').replace('``', '')
            clean_original = clean_original.replace('`sql', '').replace('`', '')
            if clean_original.lower().startswith('sql '):
                clean_original = clean_original[4:]
            clean_original = clean_original.strip()
            
            # Show cleaned original generated SQL for debugging
            st.write(f"**Original Generated SQL:** `{clean_original}`")
            
            # Validate and fix the SQL query
            fixed_sql = validate_and_fix_sql(generated_sql, table_sql_name)
            
            # Show if any changes were made
            if fixed_sql != clean_original:
                st.write(f"**Fixed SQL Query:** `{fixed_sql}`")
                st.success("✅ Table name was automatically corrected!")
            else:
                st.write(f"**SQL Query:** `{fixed_sql}`")
            
            st.write(f"**Using table:** `{table_sql_name}`")

            # Execute the query against the SQL Server database
            st.write("Running query against the database...")
            
            query_result, column_names = read_sql_query(fixed_sql, conn_params)

            # Display the results of the query
            st.subheader("Results:")
            if query_result:
                # Create a more structured table output
                if "Error" not in column_names:
                    try:
                        import pandas as pd
                        
                        # Convert pyodbc Row objects to lists if needed
                        data_as_lists = [list(row) for row in query_result]
                        
                        # Create DataFrame with proper error handling for column count mismatch
                        if len(data_as_lists) > 0:
                            # Check if columns match data dimensions
                            if len(column_names) == len(data_as_lists[0]):
                                df = pd.DataFrame(data_as_lists, columns=column_names)
                                st.dataframe(df)
                            else:
                                st.error(f"Column count mismatch: {len(column_names)} columns provided for data with {len(data_as_lists[0])} columns")
                                # Display without column names as fallback
                                df = pd.DataFrame(data_as_lists)
                                st.dataframe(df)
                        else:
                            st.write("Query executed successfully but returned no data.")
                    except Exception as e:
                        st.error(f"Error displaying results: {str(e)}")
                        st.write("Raw Results:")
                        for row in query_result:
                            st.write(row)
                else:
                    st.error(query_result[0][0])
            else:
                st.write("No results found.")

# Basic styling for the app
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
        }
        .css-1d391kg {
            background-color: #f9f9f9;
        }
    </style>
    """, unsafe_allow_html=True
)