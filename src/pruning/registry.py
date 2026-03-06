from typing import Callable, Dict

from .magnitude import MagnitudePruning

PRUNER_REGISTRY: Dict[str, Callable[[], object]] = {
    "magnitude": MagnitudePruning
}


def get_pruner(method: str):
    method = method.lower().strip()
    if method not in PRUNER_REGISTRY:
        raise ValueError(f"Pruning method '{method}' is not registered. Available methods: {list(PRUNER_REGISTRY.keys())}")
    return PRUNER_REGISTRY[method]()