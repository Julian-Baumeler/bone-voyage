"""CLI entrypoints."""

from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(help="Bone Voyage / OpenGEM — CT to FE models (local web app)")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (default localhost only)"),
    port: int = typer.Option(8742, help="Port"),
    reload: bool = typer.Option(False, help="Dev auto-reload"),
) -> None:
    """Start the Bone Voyage / OpenGEM local compute engine."""
    typer.echo(f"Bone Voyage engine → http://{host}:{port}")
    typer.echo("Keep this running. The website (GitHub Pages or local UI) will connect.")
    typer.echo("UI: https://julian-baumeler.github.io/bone-voyage/")
    typer.echo("Research use only. Not for diagnosis or treatment.")
    uvicorn.run("opengem.api.app:app", host=host, port=port, reload=reload)


@app.command()
def version() -> None:
    from opengem import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
