"""
pytest configuration: ensures the repository root is importable so that
`import gnn_poset_rank` resolves correctly whether or not the package has
been installed with `pip install -e .`.

This is a defensive addition, not a substitute for installing the
package properly (installing is still the documented, recommended path
in the README, and is required to use the package -- e.g. `import
gnn_poset_rank` -- outside of the tests/ directory or the scripts/
directory, both of which are run from the repository root). Its purpose
is narrower: to avoid a confusing ModuleNotFoundError specifically when
running the bare `pytest` command (as opposed to `python -m pytest`,
which already prepends the current directory to sys.path automatically)
before installing, which is an easy step to miss and produces an error
that does not obviously point back to a missing `pip install -e .`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
