import json
import re
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agent.dspy_signatures import Router, NLToSQL, SQLRepair, Synthesizer
from agent.rag.retrieval import Retriever
from agent.tools.sqlite_tool import SQLiteTool

class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    retrieved_docs: List[Dict]
    sql: str
    sql_results: Dict
    final_answer: Any
    explanation: str
    citations: List[str]
    confidence: float
    repair_count: int
    trace: List[str]

class HybridAgent:
    def __init__(self, db_path: str, docs_path: str):
        self.db_tool = SQLiteTool(db_path)
        self.retriever = Retriever(docs_path)
        self.schema = self.db_tool.get_schema()
        
        # DSPy modules
        self.router = Router()
        self.nl_to_sql = NLToSQL()
        self.sql_repair = SQLRepair()
        self.synthesizer = Synthesizer()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph with 6+ nodes and repair loop."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("nl_to_sql", self._nl_to_sql_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("repair", self._repair_node)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "planner")
        
        # Conditional from planner
        workflow.add_conditional_edges(
            "planner",
            self._should_query_sql,
            {
                "sql": "nl_to_sql",
                "rag_only": "synthesizer"
            }
        )
        
        workflow.add_edge("nl_to_sql", "executor")
        
        # Conditional from executor
        workflow.add_conditional_edges(
            "executor",
            self._should_repair,
            {
                "repair": "repair",
                "synthesize": "synthesizer"
            }
        )
        
        workflow.add_edge("repair", "executor")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _route_node(self, state: AgentState) -> AgentState:
        """Node 1: Route the question."""
        route = self.router.forward(state["question"])
        state["route"] = route
        state["trace"].append(f"Routed to: {route}")
        return state
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant documents."""
        results = self.retriever.search(state["question"], top_k=3)
        state["retrieved_docs"] = results
        state["trace"].append(f"Retrieved {len(results)} docs")
        return state
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Node 3: Extract constraints and plan approach."""
        # Extract date ranges, categories, KPI formulas from docs
        doc_context = "\n".join([doc['content'] for doc in state["retrieved_docs"]])
        
        # Simple extraction patterns
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', doc_context)
        state["trace"].append(f"Planner: Found {len(dates)} dates, route={state['route']}")
        return state
    
    def _should_query_sql(self, state: AgentState) -> str:
        """Decide if we need SQL."""
        if state["route"] in ["sql", "hybrid"]:
            return "sql"
        return "rag_only"
    
    def _nl_to_sql_node(self, state: AgentState) -> AgentState:
        """Node 4: Generate SQL query."""
        doc_context = "\n".join([doc['content'] for doc in state["retrieved_docs"]])
        
        sql = self.nl_to_sql.forward(
            question=state["question"],
            schema=self.schema,
            context=doc_context
        )
        
        state["sql"] = sql
        state["trace"].append(f"Generated SQL: {sql[:100]}...")
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Node 5: Execute SQL query."""
        if not state.get("sql"):
            state["sql_results"] = {"success": True, "rows": [], "columns": []}
            return state
        
        result = self.db_tool.execute_query(state["sql"])
        state["sql_results"] = result
        
        if result["success"]:
            state["trace"].append(f"SQL success: {len(result['rows'])} rows")
        else:
            state["trace"].append(f"SQL error: {result['error']}")
        
        return state
    
    def _should_repair(self, state: AgentState) -> str:
        """Decide if repair is needed."""
        if state.get("repair_count", 0) >= 2:
            return "synthesize"
        
        if not state["sql_results"]["success"]:
            return "repair"
        
        return "synthesize"
    
    def _repair_node(self, state: AgentState) -> AgentState:
        """Node 7: Repair failed SQL."""
        state["repair_count"] = state.get("repair_count", 0) + 1
        
        fixed_sql = self.sql_repair.forward(
            original_sql=state["sql"],
            error=state["sql_results"]["error"],
            schema=self.schema
        )
        
        state["sql"] = fixed_sql
        state["trace"].append(f"Repair attempt {state['repair_count']}: {fixed_sql[:100]}...")
        return state
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Node 6: Synthesize final answer."""
        doc_context = "\n".join([
            f"{doc['id']}: {doc['content']}" 
            for doc in state["retrieved_docs"]
        ])
        
        sql_results_str = json.dumps(state["sql_results"]["rows"], indent=2) if state["sql_results"]["success"] else ""
        
        answer_raw, explanation = self.synthesizer.forward(
            question=state["question"],
            format_hint=state["format_hint"],
            doc_context=doc_context,
            sql_results=sql_results_str
        )
        
        # Parse answer based on format_hint
        final_answer = self._parse_answer(answer_raw, state["format_hint"])
        
        # Extract citations
        citations = self._extract_citations(state)
        
        # Calculate confidence
        confidence = self._calculate_confidence(state)
        
        state["final_answer"] = final_answer
        state["explanation"] = explanation
        state["citations"] = citations
        state["confidence"] = confidence
        state["trace"].append(f"Synthesized answer: {final_answer}")
        
        return state
    
    def _parse_answer(self, answer: str, format_hint: str) -> Any:
        """Parse answer to match format_hint."""
        answer = answer.strip()
        
        # Remove markdown code blocks
        if '```' in answer:
            answer = answer.split('```')[1]
            if answer.startswith('json\n'):
                answer = answer[5:]
            answer = answer.strip()
        
        try:
            if format_hint == "int":
                return int(re.search(r'\d+', answer).group())
            elif format_hint == "float":
                match = re.search(r'[\d.]+', answer)
                return round(float(match.group()), 2) if match else 0.0
            elif "list[" in format_hint:
                # Try to parse as JSON
                return json.loads(answer)
            elif "{" in format_hint:
                # Parse as dict
                return json.loads(answer)
            else:
                return answer
        except:
            # Fallback
            try:
                return json.loads(answer)
            except:
                return answer
    
    def _extract_citations(self, state: AgentState) -> List[str]:
        """Extract citations from docs and SQL."""
        citations = []
        
        # Add doc chunks
        for doc in state["retrieved_docs"]:
            if doc['score'] > 0.1:  # Only cite relevant docs
                citations.append(doc['id'])
        
        # Add SQL tables
        if state["sql_results"]["success"] and state.get("sql"):
            sql = state["sql"].upper()
            tables = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]
            for table in tables:
                if table.upper() in sql or f'"{table.upper()}"' in sql:
                    citations.append(table)
        
        return list(set(citations))
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score."""
        confidence = 0.5
        
        # Boost for successful SQL
        if state["sql_results"]["success"] and state["sql_results"]["rows"]:
            confidence += 0.3
        
        # Boost for good retrieval scores
        if state["retrieved_docs"]:
            avg_score = sum(d['score'] for d in state["retrieved_docs"]) / len(state["retrieved_docs"])
            confidence += avg_score * 0.2
        
        # Penalize repairs
        confidence -= state.get("repair_count", 0) * 0.1
        
        return round(max(0.0, min(1.0, confidence)), 2)
    
    def run(self, question: str, format_hint: str) -> Dict[str, Any]:
        """Run the agent on a question."""
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "route": "",
            "retrieved_docs": [],
            "sql": "",
            "sql_results": {},
            "final_answer": None,
            "explanation": "",
            "citations": [],
            "confidence": 0.0,
            "repair_count": 0,
            "trace": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"]
        }