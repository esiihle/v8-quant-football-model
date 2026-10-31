#!/usr/bin/env python3
"""
Run the exploration notebook top to bottom on the real data, and save it.

Why a script [added 29 Oct 2026]
--------------------------------
The notebook's saved outputs are part of what GitHub shows a reader, so they
are held by a test (`test_the_notebook_was_run_on_real_data_with_the_current_config`):
every code cell executed, no errors, real data, the current config's
fingerprint. The first commit after that test arrived failed CI because
"Restart, Run All, then save" in an editor had left no outputs in the file.
Nothing was wrong except a manual step, and a manual step is the one place a
reproducible pipeline can quietly stop being reproducible.

This does the same thing as Restart & Run All, from the command line:

* a FRESH kernel, so nothing from an earlier session leaks into the outputs;
* cells run in order, and the first error stops the run and saves NOTHING;
* run from notebooks/, exactly as an editor would.

Usage
-----
    python scripts/run_notebook.py                     # notebooks/01_exploration.ipynb
    python scripts/run_notebook.py --check             # run it, but do not save

It refuses to run without the real data: a notebook executed on the bundled
sample would be saved with sample numbers, which is the fault it exists to stop.
"""
import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import ingest, logging_setup

NOTEBOOK = REPO_ROOT / "notebooks" / "01_exploration.ipynb"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--notebook", default=str(NOTEBOOK))
    parser.add_argument("--check", action="store_true",
                        help="execute, but leave the file on disk unchanged")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds allowed per cell (default 900)")
    args = parser.parse_args()

    # --- Refuse before doing any work if the real data is not all there -----
    config = load_config(REPO_ROOT / "config.yaml")
    missing = [entry["path"] for entry in ingest.expected_raw_files(config)
               if not entry["exists"]]
    if missing:
        print("REFUSED - the notebook would run on the bundled sample. Missing:")
        for path in missing:
            print(f"  {path}")
        print("Rebuild them with scripts/rebuild_raw_from_mirrors.py first.")
        return 1

    path = Path(args.notebook)
    notebook = nbformat.read(path, as_version=4)
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Running {path.name}: {code_cells} code cells, fresh kernel, "
          f"config fingerprint {logging_setup.run_fingerprint(config)}")

    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as error:
        # Saving a half-run notebook would publish a mix of fresh and missing
        # outputs. Stop with the error and leave the file as it was.
        print("\nFAILED - a cell raised an error; the notebook was NOT saved.\n")
        print(str(error)[:2000])
        return 1

    executed = sum(bool(cell.get("outputs")) for cell in notebook.cells
                   if cell.cell_type == "code")
    print(f"  all {code_cells} code cells ran; {executed} produced output")

    # Echo the first cell's output: the fingerprint and data source the test
    # will look for.
    first = next(cell for cell in notebook.cells if cell.cell_type == "code")
    for output in first.get("outputs", []):
        if output.get("output_type") == "stream":
            print("  " + output["text"].strip().replace("\n", "\n  "))

    if args.check:
        print("\n--check: executed successfully, file left unchanged.")
        return 0
    nbformat.write(notebook, path)
    print(f"\nSaved {path.relative_to(REPO_ROOT)} with fresh outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
