import logging

import typer

from ai_unit_test.cli import app

DEFAULT_VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")


@app.callback()
def main(verbose: bool = DEFAULT_VERBOSE_OPTION) -> None:
    """
    AI Unit Test
    """
    log_level: int = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )
    logging.debug(f"log_level: {log_level}")


if __name__ == "__main__":
    app()
