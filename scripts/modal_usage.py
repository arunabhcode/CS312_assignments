"""Print Modal billing usage for one environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

from utils import config_str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Modal usage for an environment."
    )
    parser.add_argument(
        "--env",
        default=config_str(
            "CONFIG_MODAL_ENVIRONMENT",
            "DL_ALCHEMY_MODAL_ENVIRONMENT",
            "MODAL_ENVIRONMENT",
        ),
        help=(
            "Modal environment to inspect. Defaults to utils.py, then "
            "Modal's active profile or workspace default."
        ),
    )
    parser.add_argument(
        "--for",
        dest="billing_range",
        default="this month",
        help=(
            "Billing range. Examples: 'today', 'this week', 'this month', "
            "'last month', or an ISO month for summary mode."
        ),
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use Modal's detailed billing report with resource breakdowns.",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print Modal's raw JSON response.",
    )
    return parser.parse_args(argv)


def run_modal_billing_command(args: argparse.Namespace) -> dict | list:
    if args.detailed:
        command = [
            sys.executable,
            "-m",
            "modal",
            "environment",
            "billing",
            "report",
            "--json",
            "--show-resources",
            "--for",
            args.billing_range,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "modal",
            "environment",
            "billing",
            "summary",
            "--json",
            "--for",
            args.billing_range,
        ]
    if args.env:
        command.append(args.env)

    result = subprocess.run(command, capture_output=True, check=True, text=True)
    return json.loads(result.stdout)


def money(value: object) -> str:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    return f"${amount:.4f}"


def print_summary(payload: Mapping[str, object], args: argparse.Namespace) -> None:
    env_text = args.env or "Modal default environment"
    print(f"Modal usage for {env_text}")
    print(f"Billing range: {args.billing_range}")
    print(f"Total metered cost: {money(payload.get('metered_cost', '0'))}")

    breakdown = payload.get("metered_cost_breakdown", {})
    if isinstance(breakdown, Mapping) and breakdown:
        print("Breakdown:")
        for name, value in sorted(breakdown.items()):
            print(f"  {name}: {money(value)}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_modal_billing_command(args)
    if args.raw_json or args.detailed:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not isinstance(payload, Mapping):
        raise SystemExit(f"Expected a JSON object from Modal, got {type(payload).__name__}.")
    print_summary(payload, args)


if __name__ == "__main__":
    main()
