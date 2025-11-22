#!/usr/bin/env python3
import click
import json
import dspy
import time
from rich.console import Console
from rich.progress import track
from agent.graph_hybrid import HybridAgent

console = Console()

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch: str, out: str):
    """Run the hybrid retail analytics agent on a batch of questions."""
    
    console.print("[bold blue]Initializing Retail Analytics Copilot...[/bold blue]")
    
    # Configure DSPy with Ollama
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
    
    console.print("[green]✓[/green] Agent initialized")
    
    # Load questions
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    console.print(f"[bold]Processing {len(questions)} questions...[/bold]\n")
    
    # Clear output file at start
    with open(out, 'w') as f:
        pass  # Create empty file
    
    # Process each question
    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold cyan]Question {i}/{len(questions)}:[/bold cyan]")
        console.print(f"  {q['question'][:100]}...")
        
        start_time = time.time()
        
        try:
            result = agent.run(q['question'], q.get('format_hint', 'str'))
            
            elapsed = time.time() - start_time
            
            output = {
                "id": q['id'],
                "final_answer": result['final_answer'],
                "sql": result.get('sql', ''),
                "confidence": result['confidence'],
                "explanation": result['explanation'],
                "citations": result['citations']
            }
            
            # SAVE IMMEDIATELY AFTER EACH QUESTION
            with open(out, 'a') as f:  # 'a' for append mode
                f.write(json.dumps(output) + '\n')
            
            console.print(f"[green]✓[/green] Answer: {result['final_answer']}")
            console.print(f"  Time: {elapsed:.1f}s")
            console.print(f"  Confidence: {result['confidence']}")
            console.print(f"  Citations: {', '.join(result['citations'][:3])}...")
            console.print(f"[dim]  Saved to {out}[/dim]")
            
        except KeyboardInterrupt:
            console.print(f"\n[yellow]⚠ Interrupted by user[/yellow]")
            console.print(f"[green]✓ Partial results saved to {out}[/green]")
            break
            
        except Exception as e:
            elapsed = time.time() - start_time
            console.print(f"[red]✗[/red] Error after {elapsed:.1f}s: {str(e)}")
            
            # Save error case too
            output = {
                "id": q['id'],
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            }
            
            with open(out, 'a') as f:
                f.write(json.dumps(output) + '\n')
    
    console.print(f"\n[bold green]✓ All results saved to {out}[/bold green]")

if __name__ == '__main__':
    main()