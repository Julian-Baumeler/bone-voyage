"""CLI entrypoints."""

from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(help="OpenGEM — CT to FE models (web successor to MITK-GEM)")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (default localhost only)"),
    port: int = typer.Option(8742, help="Port"),
    reload: bool = typer.Option(False, help="Dev auto-reload"),
) -> None:
    """Start the OpenGEM web server."""
    typer.echo(f"OpenGEM → http://{host}:{port}")
    typer.echo("Research use only. Not for diagnosis or treatment.")
    uvicorn.run("opengem.api.app:app", host=host, port=port, reload=reload)


@app.command()
def version() -> None:
    from opengem import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
