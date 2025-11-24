import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from main import RetailInsightsAssistant, AgentConfig, RetailDataManager

st.set_page_config(
    page_title="Retail Insights Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_assistant(data_path, api_key):
    config = AgentConfig(
        llm_model="gpt-4",
        temperature=0.1,
        api_key=api_key
    )
    return RetailInsightsAssistant(config=config, data_path=data_path)

def display_chat_message(role, content, timestamp=None):
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    st.markdown(f"""
        <div class="chat-message {css_class}">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <strong>{role.capitalize()}</strong>
                {f'<span style="color: #666; font-size: 0.8rem; margin-left: auto;">{timestamp}</span>' if timestamp else ''}
            </div>
            <div>{content}</div>
        </div>
    """, unsafe_allow_html=True)

def display_data_overview(data_manager):
    st.sidebar.markdown("### 📊 Data Overview")
    if data_manager.data_loaded:
        for table_name, metadata in data_manager.metadata.items():
            with st.sidebar.expander(f"Table: {table_name}"):
                st.write(f"**Rows:** {metadata['row_count']:,}")
                st.write(f"**Columns:** {len(metadata['columns'])}")
                st.write("**Schema:**")
                for col in metadata['columns'][:5]:
                    st.text(f"  • {col}")
                if len(metadata['columns']) > 5:
                    st.text(f"  ... and {len(metadata['columns']) - 5} more")

def create_sample_visualizations(data_manager):
    try:
        table_name = list(data_manager.metadata.keys())[0]
        query = f"SELECT * FROM {table_name} LIMIT 1000"
        df = data_manager.execute_query(query)
        if not df.empty:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                agg_data = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().reset_index()
                fig = px.bar(
                    agg_data.head(10),
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=f"{numeric_cols[0]} by {categorical_cols[0]}",
                    color=numeric_cols[0],
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Visualizations will appear here based on your queries")

def main():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #1976D2;'> Retail Insights Assistant</h1>
            <p style='font-size: 1.2rem; color: #666;'>
                AI-Powered Multi-Agent System for Retail Analytics
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        st.markdown("### Data Upload")
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True
        )
        data_path = st.text_input(
            "Or enter data directory path",
            value="./data",
            help="Path to directory containing CSV files"
        )
        if st.button("Initialize Assistant", type="primary"):
            if not api_key:
                st.error("Please provide an OpenAI API key")
            else:
                with st.spinner("Initializing assistant..."):
                    try:
                        if uploaded_files:
                            os.makedirs("./temp_data", exist_ok=True)
                            for file in uploaded_files:
                                with open(f"./temp_data/{file.name}", "wb") as f:
                                    f.write(file.getvalue())
                            data_path = "./temp_data"
                        st.session_state.assistant = initialize_assistant(data_path, api_key)
                        st.session_state.initialized = True
                        st.success("Assistant initialized successfully!")
                    except Exception as e:
                        st.error(f"Initialization error: {str(e)}")
        st.markdown("---")
        if hasattr(st.session_state, 'assistant'):
            display_data_overview(st.session_state.assistant.data_manager)
        st.markdown("---")
        st.markdown("### Mode")
        mode = st.radio(
            "Select mode:",
            ["Conversational Q&A", "Summarization"],
            help="Choose between asking specific questions or generating summaries"
        )
        st.session_state.mode = mode
        st.markdown("### Quick Actions")
        if st.button("Clear Chat History"):
            if 'messages' in st.session_state:
                st.session_state.messages = []
                st.rerun()
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.initialized:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>Query Understanding</h3>
                    <p>Natural language processing to understand your analytical questions</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>Multi-Agent System</h3>
                    <p>Coordinated agents for query resolution, data extraction, and validation</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>Instant Insights</h3>
                    <p>Real-time analysis and visualization of your retail data</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Example Queries")
        examples = [
            "Which category saw the highest YoY growth in Q3 in the North region?",
            "What are the top 5 products by revenue?",
            "Compare sales performance across all regions",
            "Generate a summary of Q4 performance",
            "Which product line underperformed last quarter?",
            "Show me sales trends over time"
        ]
        cols = st.columns(2)
        for idx, example in enumerate(examples):
            with cols[idx % 2]:
                st.info(f"{example}")
        st.markdown("---")
        st.info("Configure the assistant in the sidebar to get started!")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Conversation")
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    display_chat_message(
                        message["role"],
                        message["content"],
                        message.get("timestamp")
                    )
            if st.session_state.mode == "Conversational Q&A":
                user_input = st.chat_input("Ask a question about your retail data...")
            else:
                user_input = None
                if st.button("Generate Summary Report"):
                    user_input = "Generate a comprehensive summary of the retail sales data"
            if user_input:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": timestamp
                })
                with st.spinner("Analyzing..."):
                    try:
                        response = st.session_state.assistant.process_query(user_input)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        with col2:
            st.markdown("### Quick Stats")
            if hasattr(st.session_state.assistant, 'data_manager'):
                summary = st.session_state.assistant.data_manager.get_summary_stats()
                for table_name, stats in summary.items():
                    st.metric(
                        label=f"{table_name.title()} Rows",
                        value=f"{stats.get('total_rows', 0):,}"
                    )
            st.markdown("---")
            st.markdown("### Visualizations")
            if hasattr(st.session_state.assistant, 'data_manager'):
                create_sample_visualizations(st.session_state.assistant.data_manager)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Powered by GPT-4 • LangGraph • DuckDB</p>
            <p style='font-size: 0.8rem;'>Multi-Agent Retail Analytics System</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
