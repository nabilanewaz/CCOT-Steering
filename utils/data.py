import inspect
import json

_PERMITTED_CALLER = "evaluate_final.py"


def load_test_set(path: str | None = None) -> list:
    """
    Raises RuntimeError if called from any script other than evaluate_final.py.
    This is the only function that may open test.jsonl.
    """
    caller = inspect.stack()[1].filename
    if not caller.endswith(_PERMITTED_CALLER):
        raise RuntimeError(
            f"\n\n  \u2717 load_test_set() called from: {caller}\n"
            f"  Test data may only be loaded from: {_PERMITTED_CALLER}\n"
            f"  Move your evaluation code there and run it once.\n"
        )
    if path is None:
        from utils.dataset_paths import get_test_path

        path = get_test_path()
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_train_pool(path: str | None = None) -> list:
    """Load the train pool (safe) and return a list of examples."""
    if path is None:
        from utils.dataset_paths import get_train_pool_path

        path = get_train_pool_path()
    with open(path) as f:
        return [json.loads(l) for l in f]
