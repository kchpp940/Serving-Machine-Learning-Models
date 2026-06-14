import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.warn(
    "mlflow_to_bentoml.py is deprecated. Please use bentosklearn.py instead. "
    "Example: python bentoml/bentosklearn.py --run-id <MLFLOW_RUN_ID>",
    DeprecationWarning,
    stacklevel=2,
)

from bentosklearn import import_from_mlflow, main


if __name__ == "__main__":
    main()
