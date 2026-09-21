"""Text dataset recipes.

Deliberately empty of imports. ``text.py`` loads the recipe modules themselves,
and re-exporting it here would make ``import hpmesh.datasets.text`` alone pull
in ``datasets``/``tokenizers``/``jinja2`` -- the assets the parent package keeps
off its own surface. Callers import the concrete module
(``hpmesh.datasets.text.text``) or, more usually, go through
``build_dataloader``, which imports it lazily because a run that overrides
``dataset`` never consults a recipe at all.
"""
