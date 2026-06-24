"""Main dataset creation entry point.

This module runs the current data preparation workflow for PromoLift AI.
"""

from src.data.load_hillstrom import save_processed_hillstrom


def main() -> None:
    """Create the processed datasets used by the project."""
    save_processed_hillstrom()


if __name__ == "__main__":
    main()
