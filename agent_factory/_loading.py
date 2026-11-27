"""Resolve Python references declared in an agent's config.toml."""

from importlib import import_module
from typing import Any, TypeVar

T = TypeVar("T")


def load_symbol(reference: str) -> Any:
    module, separator, attribute = reference.partition(":")
    if not separator or not module or not attribute:
        raise ValueError(f"Expected module:attribute, got {reference!r}")
    try:
        return getattr(import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot load component {reference!r}: {exc}") from exc


def construct(spec: dict[str, Any], expected_type: type[T]) -> T:
    options = spec.copy()
    ref = options.pop("ref")
    constructor = load_symbol(ref)
    instance = constructor(**options)
    if not isinstance(instance, expected_type):
        raise TypeError(f"{ref} must construct a {expected_type.__name__}")
    return instance
