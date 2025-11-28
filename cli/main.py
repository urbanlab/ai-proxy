from rich.console import Console
from rich.markdown import Markdown
import typer
import secrets
import yaml
import json
import random
import lib.openai as openai


# Load configuration
with open("/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


# load questions.json
with open("./questions.json", "r") as f:
    QUESTIONS = json.load(f)

base_url = CONFIG.get("api_base", "http://localhost:11434/v1")
mode = "sync"
model = CONFIG.get("default_model", "devstral")
apitoken = CONFIG.get("token", None)

app = typer.Typer()
benchmark = typer.Typer()
token = typer.Typer()

app.add_typer(benchmark, name="benchmark")
app.add_typer(token, name="token")

console = Console()
console.print(Markdown("# 🔲 AI Proxy CLI"))

@benchmark.command()
def completion(mode = "sync",model: str = model, n: int = 5):
    console.print(f"Benchmarking completion: {model=}, {n=}")
    if mode == "sync":
        for i in range(n):
            randomIndex = random.randint(0, len(QUESTIONS)-1)
            response = openai.completion(model, QUESTIONS[randomIndex], url=f"{base_url}", token=apitoken)
            console.print(f"Response {i+1}: {response}")
        
    else:
        console.print("Using asynchronous mode (not implemented yet)")
        return
    

@token.command()
def generate(length: int = 30):
    # generate a sk-lenghth random token (letters and digits)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    token = ''.join(secrets.choice(alphabet) for _ in range(length))
    console.print(f"Generated token: [bold green]sk-{token}[/bold green]")

if __name__ == "__main__":
    app()