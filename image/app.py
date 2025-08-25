import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import trino.dbapi
import os

st.set_page_config(
    page_title="ICRC Missing Persons Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ICRC Missing Persons Field Report Dashboard")

@st.cache_data
def load_data(file):
    """Load and process the CSV data"""
    df = pd.read_csv(file)
    df['Update Date'] = pd.to_datetime(df['Update Date'])
    return df

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
    
    df = None
    
    if data_source == "CSV Upload":
        # File upload section
        uploaded_file = st.sidebar.file_uploader(
            "Upload Field Report Matches CSV", 
            type=['csv'],
            help="Upload a CSV file with columns: Claim ID, Missing Person Name, Update Date, Description, Field Report Ref"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.sidebar.success("✅ Custom CSV loaded successfully!")
        else:
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
                            st.sidebar.success(f"✅ Loaded {len(df)} records from {selected_table}")
                        else:
                            st.sidebar.error("❌ Failed to load table data")
    
    if df is None:
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
    tab1, tab2, tab3 = st.tabs(["📈 Latest Updates", "🔍 Search & History", "📊 Analytics"])
    
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
                selected_value1 = st.selectbox(f"Search by {search_col1}", values1, key="search1")
            
            with col2:
                # Second search column
                search_col2 = search_columns[1] if len(search_columns) > 1 else search_columns[0]
                values2 = ['All'] + sorted(df[search_col2].dropna().unique().tolist())
                selected_value2 = st.selectbox(f"Search by {search_col2}", values2, key="search2")
            
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
                    fig_cat = px.bar(
                        x=counts.values,
                        y=counts.index,
                        orientation='h',
                        title=f"Top 10 {cat_col} Counts",
                        labels={'x': 'Count', 'y': cat_col}
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
                    fig_time = px.bar(
                        x=time_counts.index.astype(str),
                        y=time_counts.values,
                        title=f"Records by Month ({date_col})",
                        labels={'x': 'Month', 'y': 'Count'}
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

if __name__ == "__main__":
    main()