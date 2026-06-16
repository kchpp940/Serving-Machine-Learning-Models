import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from car_pricing_api.app import app, create_app

__all__ = ["app", "create_app"]
