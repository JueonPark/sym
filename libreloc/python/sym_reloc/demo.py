"""Run installed examples and fail if a required numerical witness fails."""

import argparse

from .doctor import diagnose, print_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = diagnose(require_cuda=args.device == "cuda", require_torch=True)
    if report["status"] == "ok":
        from .examples import layout, typed, weights

        scenarios = [("layout", layout), ("typed", typed), ("weights", weights)]
        if args.device == "cuda":
            from .examples import transfers

            scenarios.append(("transfers", transfers))
        for name, scenario in scenarios:
            try:
                detail = scenario.run(args.device)
                report["checks"][name] = dict(status="ok", detail=detail)
            except Exception as error:
                report["status"] = "failed"
                report["checks"][name] = dict(status="failed", detail=str(error))
    print_report(report, args.json)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
