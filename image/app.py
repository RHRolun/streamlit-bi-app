import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import trino.dbapi
import os
import uuid
from typing import List

# RAG imports (optional - graceful degradation if not available)
try:
    from llama_stack_client import LlamaStackClient, RAGDocument
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("RAG functionality not available. Install llama-stack-client to enable chat features.")

st.set_page_config(
    page_title="ICRC Missing Persons Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ICRC Missing Persons Field Report Dashboard")

@st.cache_data
def load_data(file):
    """Load and process the CSV data"""
    try:
        # Try reading with error handling for malformed lines
        df = pd.read_csv(file, on_bad_lines='skip', engine='python')
        
        # Try to find and convert date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue
        
        return df
        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of failing

@st.cache_data
def get_starburst_tables():
    """Get list of available tables from Starburst"""
    try:
        starburst_host = os.getenv("STARBURST_HOST", "starburst-fabricdemo.apps.cluster-v42lf.v42lf.sandbox239.opentlc.com")
        conn = trino.dbapi.connect(
            host=starburst_host,
            port=443,
            user="admin",
            http_scheme="https",
            verify=False,
            catalog="system"
        )
        
        cursor = conn.cursor()
        cursor.execute("SHOW CATALOGS")
        catalogs = [row[0] for row in cursor.fetchall()]
        
        tables = []
        for catalog in catalogs:
            try:
                cursor.execute(f"SHOW SCHEMAS FROM {catalog}")
                schemas = [row[0] for row in cursor.fetchall()]
                
                for schema in schemas:
                    try:
                        cursor.execute(f"SHOW TABLES FROM {catalog}.{schema}")
                        schema_tables = cursor.fetchall()
                        for table in schema_tables:
                            tables.append(f"{catalog}.{schema}.{table[0]}")
                    except Exception:
                        continue
            except Exception:
                continue
        
        cursor.close()
        conn.close()
        return sorted(tables)
    
    except Exception as e:
        st.error(f"Error connecting to Starburst: {str(e)}")
        return []

@st.cache_data
def load_starburst_table(table_name):
    """Load data from Starburst table"""
    try:
        starburst_host = os.getenv("STARBURST_HOST", "starburst-fabricdemo.apps.cluster-v42lf.v42lf.sandbox239.opentlc.com")
        conn = trino.dbapi.connect(
            host=starburst_host,
            port=443,
            user="admin",
            http_scheme="https",
            verify=False,
            catalog="system"
        )
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        
        # Try to find and convert date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue
        
        conn.close()
        return df
    
    except Exception as e:
        st.error(f"Error loading table from Starburst: {str(e)}")
        return None

def main():
    st.sidebar.header("Data Source")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["CSV Upload", "Starburst Database"],
        help="Select whether to upload a CSV file or connect to Starburst database"
    )
    
    # Initialize session state for data persistence
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    if 'data_source_type' not in st.session_state:
        st.session_state.data_source_type = None
    
    df = st.session_state.loaded_data
    
    if data_source == "CSV Upload":
        # File upload section
        uploaded_file = st.sidebar.file_uploader(
            "Upload Field Report Matches CSV", 
            type=['csv'],
            help="Upload a CSV file with columns: Claim ID, Missing Person Name, Update Date, Description, Field Report Ref"
        )
        
        if uploaded_file is not None:
            # Only reload if it's a different file or no data is cached
            if (st.session_state.loaded_data is None or 
                st.session_state.data_source_type != 'csv' or 
                'uploaded_file_name' not in st.session_state or 
                st.session_state.uploaded_file_name != uploaded_file.name):
                
                df = load_data(uploaded_file)
                st.session_state.loaded_data = df
                st.session_state.data_source_type = 'csv'
                st.session_state.uploaded_file_name = uploaded_file.name
                st.sidebar.success("✅ Custom CSV loaded successfully!")
            else:
                df = st.session_state.loaded_data
                st.sidebar.success("✅ Custom CSV loaded successfully!")
        else:
            if st.session_state.data_source_type == 'csv':
                # Clear cached data if no file is uploaded
                st.session_state.loaded_data = None
                st.session_state.data_source_type = None
                df = None
            st.error("Please upload a CSV file to continue.")
            return
    
    elif data_source == "Starburst Database":
        st.sidebar.subheader("Starburst Connection")
        starburst_host = os.getenv("STARBURST_HOST", "starburst-fabricdemo.apps.cluster-v42lf.v42lf.sandbox239.opentlc.com")
        st.sidebar.info(f"Connecting to: {starburst_host}")
        
        if st.sidebar.button("Connect & Load Tables"):
            with st.spinner("Connecting to Starburst and loading table list..."):
                tables = get_starburst_tables()
                if tables:
                    st.session_state.starburst_tables = tables
                    st.sidebar.success(f"✅ Connected! Found {len(tables)} tables")
                else:
                    st.sidebar.error("❌ Connection failed or no tables found")
        
        if 'starburst_tables' in st.session_state:
            selected_table = st.sidebar.selectbox(
                "Select Table:",
                options=[None] + st.session_state.starburst_tables,
                format_func=lambda x: "Choose a table..." if x is None else x
            )
            
            if selected_table:
                if st.sidebar.button("Load Selected Table"):
                    with st.spinner(f"Loading data from {selected_table}..."):
                        df = load_starburst_table(selected_table)
                        if df is not None:
                            # Store in session state for persistence
                            st.session_state.loaded_data = df
                            st.session_state.data_source_type = 'starburst'
                            st.session_state.selected_table = selected_table
                            st.sidebar.success(f"✅ Loaded {len(df)} records from {selected_table}")
                        else:
                            st.sidebar.error("❌ Failed to load table data")
                elif 'selected_table' in st.session_state and st.session_state.selected_table == selected_table and st.session_state.loaded_data is not None:
                    # Data already loaded for this table
                    df = st.session_state.loaded_data
                    st.sidebar.success(f"✅ Using cached data: {len(df)} records from {selected_table}")
    
    if df is None or df.empty:
        st.info("Please select and load a data source to continue.")
        return
    
    # Display basic stats
    st.sidebar.markdown("### Data Overview")
    st.sidebar.metric("Total Records", len(df))
    
    # Try to show metrics for expected columns, but handle cases where they might not exist
    if 'Claim ID' in df.columns:
        st.sidebar.metric("Unique Claims", df['Claim ID'].nunique())
    if 'Missing Person Name' in df.columns:
        st.sidebar.metric("Unique Persons", df['Missing Person Name'].nunique())
    
    # Show available columns
    st.sidebar.markdown("### Available Columns")
    st.sidebar.write(list(df.columns))
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Latest Updates", "🔍 Search & History", "📊 Analytics", "💬 Chat with Data"])
    
    with tab1:
        st.header("Latest Updates")
        
        # Find date column for sorting
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col:
            # Show most recent updates
            latest_df = df.sort_values(date_col, ascending=False).head(20)
            
            st.subheader("Most Recent 20 Updates")
            st.dataframe(latest_df, use_container_width=True)
            
            # Timeline visualization if we have appropriate columns
            if len(df.columns) >= 2:
                y_col = [col for col in df.columns if col != date_col][0]
                fig_timeline = px.scatter(
                    latest_df,
                    x=date_col,
                    y=y_col,
                    title="Recent Updates Timeline"
                )
                fig_timeline.update_layout(height=500)
                st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.subheader("Data Preview (First 20 Rows)")
            st.dataframe(df.head(20), use_container_width=True)
            st.info("No date column found for timeline visualization")
    
    with tab2:
        st.header("Search & History")
        
        # Dynamic search based on available columns
        search_columns = [col for col in df.columns if df[col].dtype == 'object' or col.lower() in ['claim id', 'missing person name']]
        
        if len(search_columns) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # First search column
                search_col1 = search_columns[0]
                values1 = ['All'] + sorted(df[search_col1].dropna().unique().tolist())
                selected_value1 = st.selectbox(f"Search by {search_col1}", values1, key=f"search1_{search_col1}")
            
            with col2:
                # Second search column
                search_col2 = search_columns[1] if len(search_columns) > 1 else search_columns[0]
                values2 = ['All'] + sorted(df[search_col2].dropna().unique().tolist())
                selected_value2 = st.selectbox(f"Search by {search_col2}", values2, key=f"search2_{search_col2}")
            
            # Filter data based on selections
            filtered_df = df.copy()
            
            if selected_value1 != 'All':
                filtered_df = filtered_df[filtered_df[search_col1] == selected_value1]
            
            if selected_value2 != 'All':
                filtered_df = filtered_df[filtered_df[search_col2] == selected_value2]
            
            # Display filtered results
            if not filtered_df.empty:
                st.subheader(f"Filtered Results ({len(filtered_df)} records)")
                
                # Find date column for sorting
                date_col = None
                for col in filtered_df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_col = col
                        break
                
                if date_col:
                    history_df = filtered_df.sort_values(date_col, ascending=False)
                else:
                    history_df = filtered_df
                
                st.dataframe(history_df, use_container_width=True)
                
                # Show timeline chart if date column exists
                if date_col and len(filtered_df) > 1:
                    st.subheader("Timeline View")
                    y_col = [col for col in filtered_df.columns if col != date_col][0]
                    fig_history = px.scatter(
                        history_df.head(50),  # Limit to 50 points for performance
                        x=date_col,
                        y=y_col,
                        title="Filtered Data Timeline"
                    )
                    fig_history.update_layout(height=400)
                    st.plotly_chart(fig_history, use_container_width=True)
            else:
                st.info("No records found for the selected criteria.")
        else:
            st.subheader("Data Explorer")
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Find categorical and date columns for analysis
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if categorical_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Count by first categorical column
                if len(categorical_cols) > 0:
                    cat_col = categorical_cols[0]
                    counts = df[cat_col].value_counts().head(10)
                    
                    # Create DataFrame for plotly to avoid list/array issues
                    chart_df = pd.DataFrame({
                        'Category': counts.index,
                        'Count': counts.values
                    })
                    
                    fig_cat = px.bar(
                        chart_df,
                        x='Count',
                        y='Category',
                        orientation='h',
                        title=f"Top 10 {cat_col} Counts",
                        labels={'Count': 'Count', 'Category': cat_col}
                    )
                    fig_cat.update_layout(height=400)
                    st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                # Time-based analysis if date column exists
                if date_cols:
                    date_col = date_cols[0]
                    df_time = df.copy()
                    df_time['Period'] = df_time[date_col].dt.to_period('M')
                    time_counts = df_time.groupby('Period').size()
                    
                    # Create DataFrame for plotly
                    time_chart_df = pd.DataFrame({
                        'Month': time_counts.index.astype(str),
                        'Count': time_counts.values
                    })
                    
                    fig_time = px.bar(
                        time_chart_df,
                        x='Month',
                        y='Count',
                        title=f"Records by Month ({date_col})",
                        labels={'Month': 'Month', 'Count': 'Count'}
                    )
                    fig_time.update_layout(height=400)
                    st.plotly_chart(fig_time, use_container_width=True)
                elif numeric_cols:
                    # Show distribution of first numeric column
                    num_col = numeric_cols[0]
                    fig_hist = px.histogram(
                        df,
                        x=num_col,
                        title=f"Distribution of {num_col}",
                        nbins=20
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("Data Summary")
        
        if categorical_cols:
            # Group by analysis for categorical columns
            summary_data = {}
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                summary_data[col] = df.groupby(col).size().sort_values(ascending=False)
            
            if summary_data:
                # Create a summary table
                summary_df = pd.DataFrame({
                    'Column': list(summary_data.keys()),
                    'Unique Values': [len(data) for data in summary_data.values()],
                    'Most Common': [data.index[0] if len(data) > 0 else 'N/A' for data in summary_data.values()],
                    'Most Common Count': [data.iloc[0] if len(data) > 0 else 0 for data in summary_data.values()]
                })
                st.dataframe(summary_df, use_container_width=True)
        
        # Basic statistics for numeric columns
        if numeric_cols:
            st.subheader("Numeric Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with tab4:
        setup_chat_interface(df)

# === RAG FUNCTIONALITY ===

def create_rag_documents(df: pd.DataFrame) -> List:
    """Convert DataFrame to RAG documents"""
    if not RAG_AVAILABLE:
        return []
    
    documents = []
    
    # Group by claim/case ID if available
    id_col = None
    for col in df.columns:
        if 'id' in col.lower() or 'claim' in col.lower():
            id_col = col
            break
    
    if id_col and df[id_col].nunique() < len(df):
        # Multiple rows per ID - group them
        for case_id in df[id_col].unique():
            case_data = df[df[id_col] == case_id]
            content = f"Case ID: {case_id}\n\n"
            
            for _, row in case_data.iterrows():
                row_text = "\n".join([f"{col}: {val}" for col, val in row.items()])
                content += f"{row_text}\n\n---\n\n"
            
            documents.append(RAGDocument(
                document_id=str(case_id),
                content=content,
                metadata={"case_id": str(case_id)}
            ))
    else:
        # One document per row
        for idx, row in df.iterrows():
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(RAGDocument(
                document_id=f"row_{idx}",
                content=content,
                metadata={"row_index": idx}
            ))
    
    return documents

def setup_rag_system(documents: List) -> tuple:
    """Initialize RAG system - returns (client, vector_db_id)"""
    if not RAG_AVAILABLE or not documents:
        return None, None
    
    try:
        base_url = os.getenv("LLAMA_STACK_BASE_URL", "https://llamastack-genaiops-rag.apps.dev.rhoai.rh-aiservices-bu.com/")
        client = LlamaStackClient(base_url=base_url)
        vector_db_id = f"icrc_data_{uuid.uuid4().hex[:8]}"
        
        # Try to register vector DB and insert documents
        try:
            client.vector_dbs.register(
                vector_db_id=vector_db_id,
                embedding_model="all-MiniLM-L6-v2",
                embedding_dimension=384,
                provider_id="milvus"
            )
        except:
            pass  # May already exist or different provider needed
        
        # Insert documents
        client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512
        )
        
        return client, vector_db_id
        
    except Exception as e:
        st.error(f"RAG setup failed: {e}")
        return None, None

def query_rag_streaming(client, vector_db_id: str, query: str):
    """Query RAG system with streaming response"""
    if not client:
        yield "RAG system not available"
        return
    
    try:
        # Retrieve documents
        rag_response = client.tool_runtime.rag_tool.query(
            content=query,
            vector_db_ids=[vector_db_id]
        )
        
        # Generate streaming response
        messages = [
            {"role": "system", "content": f"You are a helpful assistant analyzing data. Use this context: {rag_response}"},
            {"role": "user", "content": query}
        ]
        
        response = client.inference.chat_completion(
            messages=messages,
            model_id=os.getenv("RAG_MODEL_ID", "llama32"),
            sampling_params={"temperature": 0.7, "max_tokens": 1000},
            stream=True
        )
        
        # Process streaming response
        for chunk in response:
            try:
                response_delta = chunk.event.delta
                if hasattr(response_delta, 'text') and response_delta.text:
                    yield response_delta.text
                elif hasattr(response_delta, 'content') and response_delta.content:
                    yield response_delta.content
            except Exception as chunk_error:
                # Handle individual chunk errors gracefully
                continue
        
    except Exception as e:
        yield f"Query failed: {e}"

def setup_chat_interface(df: pd.DataFrame):
    """Setup chat interface for the current data"""
    st.header("💬 Chat with Your Data")
    
    if not RAG_AVAILABLE:
        st.error("🚫 RAG functionality not available. Please install llama-stack-client.")
        return
    
    if df is None or df.empty:
        st.info("📋 Please load data first to enable chat functionality.")
        return
    
    # Check if RAG setup is valid for current data
    data_source = getattr(st.session_state, 'data_source_type', 'unknown')
    table_name = getattr(st.session_state, 'selected_table', 'csv_upload')
    data_id = f"{data_source}_{table_name}_{len(df)}"
    
    if 'rag_data_id' not in st.session_state:
        st.session_state.rag_data_id = None
    
    # Debug info (remove this after fixing)
    setup_done = st.session_state.get("rag_setup_done", False)
    current_data_id = st.session_state.get("rag_data_id", "none")
    
    # Step 1: Ask user to confirm setup
    if (not setup_done or current_data_id != data_id):
        
        st.subheader("🔧 Setup Required")
        st.info(f"📄 Ready to process {len(df)} rows of data for chat.")
        # Debug info
        st.write(f"Debug: setup_done={setup_done}, current_data_id={current_data_id}, new_data_id={data_id}")
        
        if st.button("🚀 Setup Chat with This Data", type="primary", key="setup_rag"):
            with st.spinner("🔄 Setting up RAG system..."):
                documents = create_rag_documents(df)
                if documents:
                    client, vector_db_id = setup_rag_system(documents)
                    if client:
                        st.session_state.rag_client = client
                        st.session_state.vector_db_id = vector_db_id
                        st.session_state.rag_setup_done = True
                        st.session_state.rag_data_id = data_id
                        st.success("✅ Chat setup complete!")
                        # Don't use st.rerun() - let it fall through to show chat interface
                    else:
                        st.error("❌ RAG setup failed")
                        return
                else:
                    st.error("❌ No documents created")
                    return
        else:
            return
    
    # Step 2: Chat interface
    st.subheader("💬 Chat Interface")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💬 Ask about your data..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in query_rag_streaming(
                st.session_state.get("rag_client"),
                st.session_state.get("vector_db_id"),
                prompt
            ):
                full_response += chunk
                response_placeholder.write(full_response + "▋")  # Add cursor effect
            
            # Remove cursor and show final response
            response_placeholder.write(full_response)
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    
    # Reset button
    if st.button("🔄 Reset Chat", key="reset_chat"):
        for key in ["chat_history", "rag_setup_done", "rag_client", "vector_db_id", "rag_data_id"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()