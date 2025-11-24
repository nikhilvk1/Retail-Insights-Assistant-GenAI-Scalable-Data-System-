import os
import json
import pandas as pd
import duckdb
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


@dataclass
class AgentConfig:
    llm_model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None


class ConversationState(TypedDict):
    user_query: str
    query_type: str
    parsed_intent: Dict[str, Any]
    sql_query: str
    query_results: Any
    validation_status: Dict[str, Any]
    final_response: str
    conversation_history: List[Dict[str, str]]
    error: Optional[str]


class RetailDataManager:
    def __init__(self, data_path: str = None):
        self.conn = duckdb.connect(':memory:')
        self.data_loaded = False
        self.metadata = {}
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        try:
            if os.path.isdir(data_path):
                csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    table_name = csv_file.replace('.csv', '').lower()
                    file_path = os.path.join(data_path, csv_file)
                    self.conn.execute(f"""
                        CREATE TABLE {table_name} AS 
                        SELECT * FROM read_csv_auto('{file_path}')
                    """)
                    self.metadata[table_name] = self._get_table_metadata(table_name)
            elif os.path.isfile(data_path):
                table_name = "sales_data"
                self.conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{data_path}')
                """)
                self.metadata[table_name] = self._get_table_metadata(table_name)
            self.data_loaded = True
            print(f"✓ Data loaded successfully. Tables: {list(self.metadata.keys())}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _get_table_metadata(self, table_name: str) -> Dict:
        schema = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return {
            'columns': schema['column_name'].tolist(),
            'types': dict(zip(schema['column_name'], schema['column_type'])),
            'row_count': row_count
        }
    
    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            raise Exception(f"Query execution error: {str(e)}")
    
    def get_summary_stats(self) -> Dict:
        if not self.data_loaded:
            return {}
        summary = {}
        for table_name in self.metadata.keys():
            try:
                stats_query = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT *) as unique_rows
                    FROM {table_name}
                """
                stats = self.conn.execute(stats_query).fetchdf()
                summary[table_name] = stats.to_dict('records')[0]
            except:
                pass
        return summary


class QueryResolutionAgent:
    def __init__(self, config: AgentConfig):
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query understanding agent for retail analytics.
            Your task is to parse natural language queries and extract structured intent.
            Return ONLY valid JSON with this structure:
            {
                "query_type": "summarization" or "qa",
                "intent": "brief description",
                "parameters": {
                    "metrics": [],
                    "filters": {},
                    "aggregations": [],
                    "comparisons": []
                }
            }"""),
            ("human", "{query}")
        ])
    
    def process(self, state: ConversationState) -> ConversationState:
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(query=state['user_query'])
            )
            parsed = json.loads(response.content)
            state['query_type'] = parsed.get('query_type', 'qa')
            state['parsed_intent'] = parsed
            print(f"✓ Query Resolution: {parsed['intent']}")
            return state
        except Exception as e:
            state['error'] = f"Query resolution error: {str(e)}"
            return state


class DataExtractionAgent:
    def __init__(self, config: AgentConfig, data_manager: RetailDataManager):
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
        )
        self.data_manager = data_manager
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL query generation agent.
            Convert intent into valid DuckDB SQL. Return ONLY SQL."""),
            ("human", "{intent}")
        ])
    
    def process(self, state: ConversationState) -> ConversationState:
        try:
            metadata_str = json.dumps(self.data_manager.metadata, indent=2)
            intent_str = json.dumps(state['parsed_intent'], indent=2)
            response = self.llm.invoke(
                self.prompt.format_messages(
                    metadata=metadata_str,
                    intent=intent_str
                )
            )
            sql_query = response.content.strip()
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            state['sql_query'] = sql_query
            results = self.data_manager.execute_query(sql_query)
            state['query_results'] = results
            print(f"✓ Data Extraction: Retrieved {len(results)} rows")
            return state
        except Exception as e:
            state['error'] = f"Data extraction error: {str(e)}"
            return state


class ValidationAgent:
    def __init__(self, config: AgentConfig):
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a validation and response generation agent.
            Return JSON:
            {
                "validation": {"status": "...", "issues": []},
                "response": "..."
            }"""),
            ("human", """User Query: {query}
            
Query Results:
{results}

Generate appropriate response.""")
        ])
    
    def process(self, state: ConversationState) -> ConversationState:
        try:
            results = state['query_results']
            if isinstance(results, pd.DataFrame):
                results_str = "No data found" if len(results) == 0 else results.to_string(index=False)
            else:
                results_str = str(results)
            response = self.llm.invoke(
                self.prompt.format_messages(
                    query=state['user_query'],
                    results=results_str
                )
            )
            parsed = json.loads(response.content)
            state['validation_status'] = parsed['validation']
            state['final_response'] = parsed['response']
            print(f"✓ Validation: {parsed['validation']['status']}")
            return state
        except Exception as e:
            state['error'] = f"Validation error: {str(e)}"
            return state


class RetailInsightsAssistant:
    def __init__(self, config: AgentConfig = None, data_path: str = None):
        self.config = config or AgentConfig()
        self.data_manager = RetailDataManager(data_path)
        
        self.query_agent = QueryResolutionAgent(self.config)
        self.extraction_agent = DataExtractionAgent(self.config, self.data_manager)
        self.validation_agent = ValidationAgent(self.config)
        
        self.workflow = self._build_workflow()
        self.conversation_history = []
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ConversationState)
        workflow.add_node("query_resolution", self.query_agent.process)
        workflow.add_node("data_extraction", self.extraction_agent.process)
        workflow.add_node("validation", self.validation_agent.process)
        workflow.set_entry_point("query_resolution")
        workflow.add_edge("query_resolution", "data_extraction")
        workflow.add_edge("data_extraction", "validation")
        workflow.add_edge("validation", END)
        return workflow.compile()
    
    def process_query(self, query: str) -> str:
        print(f"\n{'='*60}")
        print(f"Processing: {query}")
        print(f"{'='*60}")
        
        state: ConversationState = {
            'user_query': query,
            'query_type': '',
            'parsed_intent': {},
            'sql_query': '',
            'query_results': None,
            'validation_status': {},
            'final_response': '',
            'conversation_history': self.conversation_history,
            'error': None
        }
        
        result = self.workflow.invoke(state)
        
        if result.get('error'):
            response = f"I encountered an error: {result['error']}"
        else:
            response = result['final_response']
        
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def generate_summary(self) -> str:
        summary_query = "Generate a comprehensive summary of the retail sales data including key metrics, trends, and insights"
        return self.process_query(summary_query)


if __name__ == "__main__":
    assistant = RetailInsightsAssistant(
        config=AgentConfig(llm_model="gpt-4"),
        data_path="./data"
    )
    
    queries = [
        "Generate a summary of overall sales performance",
        "Which region had the highest sales in Q3?",
        "What was the year-over-year growth for Electronics?",
        "Show me the top 5 products by revenue"
    ]
    
    for query in queries:
        response = assistant.process_query(query)
        print(f"\nResponse: {response}\n")
