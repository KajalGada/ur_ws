"""
Reprocesses existing npz demo files to ensure they contain aligned
(states, actions) pairs where actions = np.diff(states, axis=0).

Safe to run on already-processed files — skips any file that already
has actions with the correct shape.
"""

import numpy as np
import glob
import os
import argparse


def reprocess(file_path: str, dry_run: bool = False) -> None:
    demo = np.load(file_path)
    states = demo["states"]

    if states.shape[0] < 2:
        print(f"SKIP  {file_path}  (fewer than 2 states)")
        return

    actions = np.diff(states, axis=0)
    states = states[:-1]

    # Check if already correct
    if "actions" in demo and demo["actions"].shape == actions.shape:
        if np.allclose(demo["actions"], actions):
            print(f"OK    {file_path}")
            return

    if dry_run:
        print(f"WOULD rewrite  {file_path}  ({len(states)} pairs)")
        return

    # Preserve any other keys (e.g. joint_names)
    extra = {k: demo[k] for k in demo.files if k not in ("states", "actions")}

    np.savez(file_path, states=states, actions=actions, **extra)
    print(f"WROTE {file_path}  ({len(states)} pairs)")


def main():
    parser = argparse.ArgumentParser(description="Reprocess dataset npz files.")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
    if not files:
        print(f"No npz files found in '{args.dataset}'")
        return

    for f in files:
        reprocess(f, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
