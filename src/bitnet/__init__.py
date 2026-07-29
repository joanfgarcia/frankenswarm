"""
src.bitnet — Backward-compatible re-exports via import hook aliasing.

After the submodule refactoring, all major symbols remain importable from
their original paths (e.g. ``from src.bitnet.modeling_bitnet import ...``)
because this package installs an import hook that redirects old module paths
to their new submodule locations.

We use a sys.meta_path finder rather than __getattr__ because Python's import
machinery resolves ``from src.bitnet.modeling_bitnet import X`` as a subpackage
lookup BEFORE __getattr__ fires, so lazy attribute aliasing alone is insufficient.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys

# Mapping: old dotted path → new dotted path
_ALIASES = {
    # model
    "src.bitnet.modeling_bitnet": "src.bitnet.model.modeling_bitnet",
    "src.bitnet.modeling_conversational": "src.bitnet.model.modeling_conversational",
    # vocab
    "src.bitnet.glyph_vocabulary": "src.bitnet.vocab.glyph_vocabulary",
    "src.bitnet.dictionary_tool": "src.bitnet.vocab.dictionary_tool",
    "src.bitnet.expand_vocabulary": "src.bitnet.vocab.expand_vocabulary",
    # translation
    "src.bitnet.translator": "src.bitnet.translation.translator",
    # telemetry — NOTE: "src.bitnet.telemetry" is NOT aliased because it
    # collides with the src/bitnet/telemetry/ subpackage. Use the new path.
    # growth
    "src.bitnet.net2net": "src.bitnet.growth.net2net",
    "src.bitnet.genetic": "src.bitnet.growth.genetic",
    "src.bitnet.consolidate_sleep": "src.bitnet.growth.consolidate_sleep",
    # data
    "src.bitnet.dataset_breeder": "src.bitnet.data.dataset_breeder",
    "src.bitnet.generalization_breeder": "src.bitnet.data.generalization_breeder",
    "src.bitnet.logic_breeder": "src.bitnet.data.logic_breeder",
    "src.bitnet.dojo_populora": "src.bitnet.data.dojo_populora",
    # operators — NOTE: "src.bitnet.operators" is NOT aliased because it
    # collides with the src/bitnet/operators/ subpackage. Use the new path.
    "src.bitnet.operators_logic": "src.bitnet.operators.operators_logic",
    # worlds
    "src.bitnet.cooperative_world": "src.bitnet.worlds.cooperative_world",
    "src.bitnet.complex_world": "src.bitnet.worlds.complex_world",
    "src.bitnet.minimal_world": "src.bitnet.worlds.minimal_world",
    "src.bitnet.puzzle_world": "src.bitnet.worlds.puzzle_world",
    "src.bitnet.prolog_world": "src.bitnet.worlds.prolog_world",
    # training
    "src.bitnet.train_sovereign_school": "src.bitnet.training.train_sovereign_school",
    # inference
    "src.bitnet.run_arena_simulation": "src.bitnet.inference.run_arena_simulation",
    "src.bitnet.run_grid_032": "src.bitnet.inference.run_grid_032",
    "src.bitnet.run_swarm_chat": "src.bitnet.inference.run_swarm_chat",
    "src.bitnet.stability_check_032": "src.bitnet.inference.stability_check_032",
}

# Also support bitnet.X (without src prefix) for some older lab scripts
_ALIASES.update({k.replace("src.bitnet.", "bitnet."): v for k, v in list(_ALIASES.items())})


class _AliasLoader(importlib.abc.Loader):
    """Loader that imports the real module and registers it under the alias."""

    def __init__(self, real_name):
        self.real_name = real_name

    def create_module(self, spec):
        return None  # Use default semantics

    def exec_module(self, module):
        real = importlib.import_module(self.real_name)
        # Replace the module in sys.modules with the real one
        sys.modules[module.__name__] = real
        module.__dict__.update(real.__dict__)


class _BitnetAliasFinder(importlib.abc.MetaPathFinder):
    """Redirect old src.bitnet.X imports to src.bitnet.submodule.X."""

    def find_spec(self, fullname, path, target=None):
        if fullname in _ALIASES:
            real_name = _ALIASES[fullname]
            loader = _AliasLoader(real_name)
            return importlib.machinery.ModuleSpec(fullname, loader)
        return None


# Install the hook once at package import time
if not any(isinstance(f, _BitnetAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _BitnetAliasFinder())
