from car_pricing.api_client import PredictionAPIClient
from car_pricing.model_runtime import CarPriceModel


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Car Price Prediction CLI")
    parser.add_argument("--api-url", default=None, help="API base URL")
    parser.add_argument("--timeout", type=int, default=None, help="Request timeout")
    parser.add_argument("--health", action="store_true", help="Check API health")
    args = parser.parse_args()

    if args.health:
        client = PredictionAPIClient(base_url=args.api_url, timeout=args.timeout)
        result = client.health_check()
        print(f"Status: {result.status}")
        print(f"Service: {result.service}")
        print(f"Version: {result.version}")
        print(f"Model Loaded: {result.model_loaded}")
    else:
        print("Car Price Prediction package is installed.")
        print("Use --help for available commands.")


if __name__ == "__main__":
    main()
