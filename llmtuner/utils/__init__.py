"""Small dependency-free helpers: no torch, no llmtuner imports above this line.

Members:

* ``gc.py`` -- ``GarbageCollection``: the collect/disable pair the trainer wraps
  around the phases where allocator churn would otherwise show up as time.
* ``logger_utils.py`` -- ``get_logger`` (a colour formatter that filters by rank
  at emit time) and ``get_distributed_rank``.
* ``lazy_exports.py`` -- the one implementation of the PEP 562 lazy package
  index that the package indexes here share.

The layer is a leaf on purpose: ``logger_utils`` reads the environment, and
``lazy_exports`` imports only the standard library, so anything can import them
without a cycle -- including ``llmtuner/__init__.py`` itself.
"""
