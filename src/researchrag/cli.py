"""Developer CLI for inspecting real retrieval and provider answers."""

import argparse
import json

from researchrag.api.dependencies import build_services
from researchrag.config import get_settings


def main() -> None:
    """Ask one question against the persisted paper library."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    settings = get_settings()
    services = build_services(settings)
    print(json.dumps(services.answers.answer(args.question).model_dump(), indent=2))


if __name__ == "__main__":
    main()
