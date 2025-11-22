#!/bin/bash

echo "=========================================="
echo "Retail Analytics Copilot - Setup Script"
echo "=========================================="

# Create directories
echo ""
echo "1. Creating directory structure..."
mkdir -p data
mkdir -p docs
mkdir -p agent/rag
mkdir -p agent/tools

# Download Northwind database
echo ""
echo "2. Downloading Northwind SQLite database..."
if [ -f "data/northwind.sqlite" ]; then
    echo "   ✓ Database already exists"
else
    curl -L -o data/northwind.sqlite \
        https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
    echo "   ✓ Database downloaded"
fi

# Create compatibility views (optional)
echo ""
echo "3. Creating database compatibility views..."
sqlite3 data/northwind.sqlite <<'SQL'
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
SQL
echo "   ✓ Views created"

# Check if Ollama is installed
echo ""
echo "4. Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "   ✓ Ollama is installed"
    
    # Check if model is pulled
    if ollama list | grep -q "phi3.5:3.8b-mini-instruct-q4_K_M"; then
        echo "   ✓ Phi-3.5 model already available"
    else
        echo "   Pulling Phi-3.5 model (this may take a few minutes)..."
        ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
        echo "   ✓ Model pulled"
    fi
else
    echo "   ✗ Ollama not found"
    echo "   Please install from: https://ollama.com"
    echo "   Then run: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
fi

# Install Python dependencies
echo ""
echo "5. Installing Python dependencies..."
pip install -q -r requirements.txt
echo "   ✓ Dependencies installed"

# Verify document files exist
echo ""
echo "6. Verifying document corpus..."
docs_ok=true
for doc in marketing_calendar.md kpi_definitions.md catalog.md product_policy.md; do
    if [ ! -f "docs/$doc" ]; then
        echo "   ✗ Missing: docs/$doc"
        docs_ok=false
    fi
done

if [ "$docs_ok" = true ]; then
    echo "   ✓ All document files present"
else
    echo "   Please create the missing document files in docs/"
fi

# Verify eval file exists
echo ""
echo "7. Verifying evaluation file..."
if [ -f "sample_questions_hybrid_eval.jsonl" ]; then
    echo "   ✓ Evaluation file exists"
else
    echo "   ✗ Missing: sample_questions_hybrid_eval.jsonl"
    echo "   Please create this file with the 6 test questions"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the agent:"
echo "  python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl"
echo ""
echo "To run DSPy optimization:"
echo "  python optimize_nl_to_sql.py"
echo ""