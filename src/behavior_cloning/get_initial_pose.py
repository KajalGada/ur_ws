"""
Reads all demo npz files and prints the initial joint pose (states[0])
from each. Useful for verifying the arm starts from a consistent position.
"""

import numpy as np
import glob
import os
import argparse


def get_initial_pose(file_path: str) -> np.ndarray:
    demo = np.load(file_path)
    return demo["states"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
    if not files:
        print(f"No npz files found in '{args.dataset}'")
        return

    for f in files:
        pose = get_initial_pose(f)
        demo = np.load(f)
        joint_names = list(demo["joint_names"]) if "joint_names" in demo else [f"j{i}" for i in range(len(pose))]
        print(f"\n{os.path.basename(f)}")
        for name, val in zip(joint_names, pose):
            print(f"  {name}: {np.degrees(val):.2f} deg  ({val:.4f} rad)")


if __name__ == "__main__":
    main()
