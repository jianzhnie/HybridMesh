"""Multimodal dataset recipes and their media helpers.

Deliberately empty of imports. Importing a module *from* here pulls in
torchvision (and, for ``mm_video``, a video backend), so a re-export would make
``import hpmesh.datasets.multimodal.mm_datasets`` -- or anything that walks this
package -- drag the whole media stack along. ``build_dataloader`` imports the
two modules it needs inside the branch that has already decided the recipe name
is multimodal, so a missing torchvision surfaces there, where it can be turned
into install guidance instead of a bare ``ModuleNotFoundError``.

Whether this module imports eagerly is not itself load-bearing: the text path
never reaches the multimodal package object, so nothing here runs for it either
way. What the guard in ``tests/unit_tests/cpu/components/data/
test_multimodal_build.py`` pins down is the property that matters -- a
text-only run leaves the media stack out of ``sys.modules`` and installable-free.
"""
