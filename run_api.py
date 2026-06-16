import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from car_pricing.config import get_config

CONFIG = get_config()


def run_fastapi():
    import uvicorn

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "car_pricing_api"))

    print(f"Starting Car Pricing API on {CONFIG.bind_address}")
    print(f"Model directory: {CONFIG.model_dir}")
    print(f"API base URL: {CONFIG.api_base_url}")

    uvicorn.run(
        "app:app",
        host=CONFIG.api_host,
        port=CONFIG.api_port,
        reload=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Car Pricing API Launcher")
    parser.add_argument(
        "--entry",
        choices=["car_pricing_api", "fastapi"],
        default="car_pricing_api",
        help="Which entry point to launch (default: car_pricing_api)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current configuration and exit",
    )
    args = parser.parse_args()

    if args.print_config:
        print("Current Runtime Configuration:")
        for k, v in CONFIG.as_dict().items():
            print(f"  {k}: {v}")
        sys.exit(0)

    if args.entry == "fastapi":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fastapi"))

    run_fastapi()
