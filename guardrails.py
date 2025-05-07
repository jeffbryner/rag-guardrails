from utils import allowed_subject, create_embedder

from rich.console import Console

console = Console()


def main():
    """Main function."""
    embedder = create_embedder()
    while True:
        prompt = console.input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        if allowed_subject(prompt, embedder=embedder):
            console.print("[green]Prompt is allowed.[/green]")
        else:
            console.print("[red]Prompt is not allowed.[/red]")


if __name__ == "__main__":
    main()
