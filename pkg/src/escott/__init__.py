"""escott: exact structured-coalescent likelihood of a fixed time tree with tensor trains, and checks of MASCOT."""
__version__ = "0.2.0"

from .api import Result, exact_method, loglik, max_lineages  # noqa: E402
from .io import (load_tree, params_from_beast_log, params_from_json, read_beast_log, read_locations_csv,  # noqa: E402
                 tree_events)

__all__ = ["loglik", "Result", "max_lineages", "exact_method", "load_tree", "tree_events", "read_locations_csv",
           "read_beast_log", "params_from_json", "params_from_beast_log"]
