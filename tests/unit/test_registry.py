from __future__ import annotations

import pytest

from ragbench.core.errors import ConfigError, RegistryError
from ragbench.core.registry import Registry


class Thing:
    def __init__(self, resources: object, size: int, label: str = "x") -> None:
        self.size = size
        self.label = label


def make_registry() -> Registry[Thing]:
    registry: Registry[Thing] = Registry("thing", "ragbench.test_things")
    registry.register("plain")(Thing)
    return registry


def test_create_builds_the_registered_factory_with_params() -> None:
    thing = make_registry().create("plain", None, {"size": 3})
    assert (thing.size, thing.label) == (3, "x")


def test_unknown_name_lists_the_allowed_names() -> None:
    with pytest.raises(RegistryError, match=r"unknown thing 'nope'; allowed: plain"):
        make_registry().get("nope")


def test_unknown_parameter_lists_the_allowed_parameters() -> None:
    with pytest.raises(
        ConfigError, match=r"does not accept colour; allowed parameters: size, label"
    ):
        make_registry().validate("plain", {"size": 1, "colour": "red"})


def test_missing_required_parameter_is_named() -> None:
    with pytest.raises(ConfigError, match="needs parameters: size"):
        make_registry().validate("plain", {})


def test_registering_a_second_factory_under_one_name_fails() -> None:
    registry = make_registry()
    registry.register("plain")(Thing)  # same factory: allowed
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("plain")(lambda resources: Thing(resources, 1))


def test_builtin_registries_know_their_implementations() -> None:
    from ragbench.core import registries

    assert {"openrouter", "hashing"} <= set(registries.embedders.names())
    assert {"openrouter", "overlap"} <= set(registries.rerankers.names())
    assert {"openrouter", "extractive"} <= set(registries.generators.names())
    assert {"memory", "postgres"} <= set(registries.caches.names())
