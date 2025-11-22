import dspy
from typing import Literal

class RouterSignature(dspy.Signature):
    """Classify the question into rag, sql, or hybrid based on what information is needed."""
    question = dspy.InputField(desc="The user's question")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")

class NLToSQLSignature(dspy.Signature):
    """Generate SQLite query from natural language question and schema."""
    question = dspy.InputField(desc="The question to answer")
    schema = dspy.InputField(desc="Database schema")
    context = dspy.InputField(desc="Additional context from documents", default="")
    sql = dspy.OutputField(desc="Valid SQLite query")

class SQLRepairSignature(dspy.Signature):
    """Fix SQL query based on error message."""
    original_sql = dspy.InputField(desc="The SQL that failed")
    error = dspy.InputField(desc="Error message")
    schema = dspy.InputField(desc="Database schema")
    fixed_sql = dspy.OutputField(desc="Corrected SQLite query")

class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from retrieved docs and SQL results."""
    question = dspy.InputField(desc="Original question")
    format_hint = dspy.InputField(desc="Expected output format")
    doc_context = dspy.InputField(desc="Retrieved document chunks", default="")
    sql_results = dspy.InputField(desc="SQL query results", default="")
    answer = dspy.OutputField(desc="Final answer matching format_hint")
    explanation = dspy.OutputField(desc="Brief explanation (1-2 sentences)")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str) -> str:
        result = self.classify(question=question)
        route = result.route.lower().strip()
        
        # Normalize to valid routes
        if 'hybrid' in route:
            return 'hybrid'
        elif 'sql' in route:
            return 'sql'
        elif 'rag' in route:
            return 'rag'
        else:
            # Default to hybrid if unclear
            return 'hybrid'

class NLToSQL(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(NLToSQLSignature)
    
    def forward(self, question: str, schema: str, context: str = "") -> str:
        result = self.generate(question=question, schema=schema, context=context)
        sql = result.sql.strip()
        
        # Clean up SQL - remove markdown code blocks if present
        if '```' in sql:
            sql = sql.split('```')[1]
            if sql.startswith('sql\n'):
                sql = sql[4:]
            sql = sql.strip()
        
        return sql

class SQLRepair(dspy.Module):
    def __init__(self):
        super().__init__()
        self.repair = dspy.ChainOfThought(SQLRepairSignature)
    
    def forward(self, original_sql: str, error: str, schema: str) -> str:
        result = self.repair(original_sql=original_sql, error=error, schema=schema)
        sql = result.fixed_sql.strip()
        
        # Clean up SQL
        if '```' in sql:
            sql = sql.split('```')[1]
            if sql.startswith('sql\n'):
                sql = sql[4:]
            sql = sql.strip()
        
        return sql

class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synth = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(self, question: str, format_hint: str, doc_context: str = "", sql_results: str = ""):
        result = self.synth(
            question=question,
            format_hint=format_hint,
            doc_context=doc_context,
            sql_results=sql_results
        )
        return result.answer.strip(), result.explanation.strip()