import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

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

def main():
    st.sidebar.header("Data Source")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader(
        "Upload Field Report Matches CSV", 
        type=['csv'],
        help="Upload a CSV file with columns: Claim ID, Missing Person Name, Update Date, Description, Field Report Ref"
    )
    
    # Load default data or uploaded data
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("✅ Custom CSV loaded successfully!")
    else:
        st.error("Sample data file not found. Please upload a CSV file.")
        return
    
    # Display basic stats
    st.sidebar.markdown("### Data Overview")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Unique Claims", df['Claim ID'].nunique())
    st.sidebar.metric("Unique Persons", df['Missing Person Name'].nunique())
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Latest Updates", "🔍 Search & History", "📊 Analytics"])
    
    with tab1:
        st.header("Latest Updates")
        
        # Show most recent updates
        latest_df = df.sort_values('Update Date', ascending=False).head(20)
        
        st.subheader("Most Recent 20 Updates")
        st.dataframe(
            latest_df[['Claim ID', 'Missing Person Name', 'Update Date', 'Description', 'Field Report Ref']],
            use_container_width=True
        )
        
        # Timeline visualization
        st.subheader("Updates Timeline")
        fig_timeline = px.scatter(
            latest_df,
            x='Update Date',
            y='Missing Person Name',
            color='Claim ID',
            hover_data=['Description', 'Field Report Ref'],
            title="Recent Updates Timeline"
        )
        fig_timeline.update_layout(height=500)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        st.header("Search & History")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Search by Claim ID
            claim_ids = ['All'] + sorted(df['Claim ID'].unique().tolist())
            selected_claim = st.selectbox("Search by Claim ID", claim_ids, key="claim_search")
        
        with col2:
            # Search by Missing Person Name
            person_names = ['All'] + sorted(df['Missing Person Name'].unique().tolist())
            selected_person = st.selectbox("Search by Missing Person Name", person_names, key="person_search")
        
        # Filter data based on selections
        filtered_df = df.copy()
        
        if selected_claim != 'All':
            filtered_df = filtered_df[filtered_df['Claim ID'] == selected_claim]
        
        if selected_person != 'All':
            filtered_df = filtered_df[filtered_df['Missing Person Name'] == selected_person]
        
        # Display filtered results
        if not filtered_df.empty:
            st.subheader(f"History ({len(filtered_df)} records)")
            
            # Sort by date for chronological view
            history_df = filtered_df.sort_values('Update Date', ascending=False)
            
            # Enhanced display with expandable details
            for idx, row in history_df.iterrows():
                with st.expander(f"🗓️ {row['Update Date'].strftime('%Y-%m-%d')} - {row['Missing Person Name']} ({row['Claim ID']})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Description:**")
                        st.write(row['Description'])
                    with col_b:
                        st.write("**Field Report Reference:**")
                        st.write(row['Field Report Ref'])
            
            # Show history as timeline chart
            if len(filtered_df) > 1:
                st.subheader("History Timeline")
                fig_history = px.line(
                    history_df.sort_values('Update Date'),
                    x='Update Date',
                    y='Claim ID',
                    color='Missing Person Name',
                    markers=True,
                    title="Status Updates Over Time"
                )
                fig_history.update_layout(height=400)
                st.plotly_chart(fig_history, use_container_width=True)
        else:
            st.info("No records found for the selected criteria.")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Updates by person
            updates_by_person = df.groupby('Missing Person Name').size().sort_values(ascending=False)
            fig_person = px.bar(
                x=updates_by_person.values,
                y=updates_by_person.index,
                orientation='h',
                title="Number of Updates by Person",
                labels={'x': 'Number of Updates', 'y': 'Missing Person Name'}
            )
            fig_person.update_layout(height=400)
            st.plotly_chart(fig_person, use_container_width=True)
        
        with col2:
            # Updates over time
            df['Update Month'] = df['Update Date'].dt.to_period('M')
            updates_by_month = df.groupby('Update Month').size()
            fig_month = px.bar(
                x=updates_by_month.index.astype(str),
                y=updates_by_month.values,
                title="Updates by Month",
                labels={'x': 'Month', 'y': 'Number of Updates'}
            )
            fig_month.update_layout(height=400)
            st.plotly_chart(fig_month, use_container_width=True)
        
        # Field reports summary
        st.subheader("Field Reports Summary")
        report_summary = df.groupby('Field Report Ref').agg({
            'Claim ID': 'nunique',
            'Missing Person Name': 'nunique',
            'Update Date': 'max'
        }).rename(columns={
            'Claim ID': 'Unique Claims',
            'Missing Person Name': 'Unique Persons',
            'Update Date': 'Latest Update'
        }).sort_values('Latest Update', ascending=False)
        
        st.dataframe(report_summary, use_container_width=True)

if __name__ == "__main__":
    main()