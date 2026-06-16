import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from car_pricing.config import get_config

CONFIG = get_config()


def run_streamlit():
    import subprocess

    streamlit_app = os.path.join(os.path.dirname(__file__), "streamlitapp", "streamlit_app.py")

    print(f"Starting Streamlit app, connecting to API at {CONFIG.api_base_url}")
    print(f"Request timeout: {CONFIG.request_timeout}s")

    env = os.environ.copy()
    env["API_BASE_URL"] = CONFIG.api_base_url
    env["REQUEST_TIMEOUT"] = str(CONFIG.request_timeout)
    env["PYTHONPATH"] = os.path.dirname(__file__)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        streamlit_app,
        "--server.address",
        CONFIG.api_host,
        "--server.port",
        str(CONFIG.api_port + 1),
    ]

    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Car Pricing Streamlit Launcher")
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

    run_streamlit()
