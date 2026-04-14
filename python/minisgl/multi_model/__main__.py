from __future__ import annotations

import argparse

from .launcher import launch_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniSGL multi-model gateway")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the multi-model JSON config file.",
    )
    args = parser.parse_args()
    launch_from_config(args.config)


if __name__ == "__main__":
    main()
