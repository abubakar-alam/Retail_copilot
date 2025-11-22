import sqlite3
from typing import Dict, List, Any, Optional

class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_schema(self) -> str:
        """Return schema information for all tables."""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_parts = []
        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info([{table}])")
            columns = cursor.fetchall()
            
            col_defs = []
            for col in columns:
                col_defs.append(f"  {col[1]} {col[2]}")
            
            schema_parts.append(f"{table}(\n" + ",\n".join(col_defs) + "\n)")
        
        return "\n\n".join(schema_parts)
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL and return results with columns and rows."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Convert rows to list of dicts
            result_rows = []
            for row in rows:
                result_rows.append(dict(zip(columns, row)))
            
            return {
                "success": True,
                "columns": columns,
                "rows": result_rows,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e)
            }
    
    def close(self):
        self.conn.close()