"""
export_tree.py
Exports ONE example tree from a trained forest as:
  1. A human-readable text trace (the actual yes/no questions it learned)
  2. A visual diagram (PNG) of that same tree

Useful for understanding/auditing what an individual tree in the forest
actually decided, and for keeping a record of a specific model version.
"""

import argparse
import joblib
from sklearn.tree import export_text, plot_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Export one tree from a trained forest")
    parser.add_argument("--model", type=str, default="../models/temp_model.joblib",
                         help="Path to a .joblib model bundle (temp_model or rain_model)")
    parser.add_argument("--tree-index", type=int, default=0,
                         help="Which tree in the forest to export (0 to n_estimators-1)")
    parser.add_argument("--max-depth", type=int, default=3,
                         help="How many levels deep to show (trees can be huge; 3 keeps it readable)")
    parser.add_argument("--out-prefix", type=str, default="../logs/tree_trace",
                         help="Output file prefix for the .txt and .png files")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    forest = bundle["model"]
    feature_names = bundle["features"]
    one_tree = forest.estimators_[args.tree_index]

    # 1. Text trace: the actual decision rules, human-readable
    text_trace = export_text(one_tree, feature_names=feature_names, max_depth=args.max_depth)
    txt_path = f"{args.out_prefix}_{args.tree_index}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Tree #{args.tree_index} of {len(forest.estimators_)} in the forest\n")
        f.write(f"(showing top {args.max_depth} levels; full tree is deeper)\n\n")
        f.write(text_trace)
    print(f"Text trace saved to {txt_path}")

    # 2. Visual diagram
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(one_tree, feature_names=feature_names, filled=True,
              max_depth=args.max_depth, fontsize=8, ax=ax, rounded=True)
    png_path = f"{args.out_prefix}_{args.tree_index}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Diagram saved to {png_path}")


if __name__ == "__main__":
    main()
