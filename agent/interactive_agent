#!/usr/bin/env python3
import dspy
from rich.console import Console
from rich.prompt import Prompt
from agent.graph_hybrid import HybridAgent
import json
import time

console = Console()

def main():
    console.print("[bold blue]🤖 Retail Analytics Copilot - Interactive Mode[/bold blue]\n")
    
    # Configure DSPy
    console.print("[yellow]Initializing (this may take a moment)...[/yellow]")
    lm = dspy.LM(
        model='ollama/phi3.5:3.8b-mini-instruct-q4_K_M',
        api_base='http://localhost:11434',
        max_tokens=300,
        temperature=0.0
    )
    dspy.configure(lm=lm)
    
    # Initialize agent
    agent = HybridAgent(
        db_path="data/northwind.sqlite",
        docs_path="docs"
    )
    
    console.print("[green]✓[/green] Ready!\n")
    console.print("[dim]Type 'quit' or 'exit' to stop[/dim]\n")
    
    # Interactive loop
    while True:
        # Get question
        question = Prompt.ask("\n[bold cyan]Your Question[/bold cyan]")
        
        if question.lower() in ['quit', 'exit', 'q']:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        
        if not question.strip():
            continue
        
        # Ask for format hint
        console.print("\n[dim]Format hint (press Enter for 'str'):[/dim]")
        console.print("[dim]Options: int, float, str, {key:type}, list[{key:type}][/dim]")
        format_hint = Prompt.ask("Format hint", default="str")
        
        # Process
        console.print(f"\n[yellow]🤔 Thinking... (this may take 2-5 minutes)[/yellow]")
        start = time.time()
        
        try:
            result = agent.run(question, format_hint)
            elapsed = time.time() - start
            
            # Display results
            console.print(f"\n[green]✅ Answer (in {elapsed:.1f}s):[/green]")
            console.print(f"[bold]{result['final_answer']}[/bold]")
            
            console.print(f"\n[cyan]📝 Explanation:[/cyan]")
            console.print(result['explanation'])
            
            console.print(f"\n[cyan]🔍 Confidence:[/cyan] {result['confidence']}")
            
            if result.get('sql'):
                console.print(f"\n[cyan]💾 SQL Used:[/cyan]")
                console.print(f"[dim]{result['sql']}[/dim]")
            
            console.print(f"\n[cyan]📚 Sources:[/cyan]")
            for citation in result['citations']:
                console.print(f"  • {citation}")
            
            # Save to history
            with open('interactive_history.jsonl', 'a') as f:
                f.write(json.dumps({
                    'question': question,
                    'format_hint': format_hint,
                    'answer': result['final_answer'],
                    'confidence': result['confidence'],
                    'sql': result.get('sql', ''),
                    'citations': result['citations']
                }) + '\n')
            
        except Exception as e:
            console.print(f"\n[red]❌ Error:[/red] {str(e)}")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

if __name__ == '__main__':
    main()