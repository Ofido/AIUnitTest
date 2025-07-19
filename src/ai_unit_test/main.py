import typer

from ai_unit_test.cli import app as cli_app

app = typer.Typer()

app.add_typer(cli_app, name="cli")

if __name__ == "__main__":
    app()
