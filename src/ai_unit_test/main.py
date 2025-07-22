import logging

import typer

from ai_unit_test.cli import app

DEFAULT_VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")


def setup_logging(verbose: bool) -> None:
    log_level: int = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )
    logging.debug(f"log_level: {log_level}")


@app.callback()
def main(verbose: bool = DEFAULT_VERBOSE_OPTION) -> None:
    """
    AI Unit Test
    """
    setup_logging(verbose)


if __name__ == "__main__":
    app()
