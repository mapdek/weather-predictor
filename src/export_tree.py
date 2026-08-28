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


def _count_leaves_at_depth(sk_tree, max_depth):
    """How many boxes will appear on the bottom row when plot_tree stops at max_depth."""
    def rec(node, depth):
        left, right = sk_tree.children_left[node], sk_tree.children_right[node]
        is_leaf = left == -1
        if depth == max_depth or is_leaf:
            return 1
        return rec(left, depth + 1) + rec(right, depth + 1)
    return rec(0, 0)


def main():
    parser = argparse.ArgumentParser(description="Export one tree from a trained forest")
    parser.add_argument("--model", type=str, default="../models/temp_model.joblib",
                         help="Path to a .joblib model bundle (temp_model or rain_model)")
    parser.add_argument("--tree-index", type=int, default=0,
                         help="Which tree in the forest to export (0 to n_estimators-1)")
    parser.add_argument("--max-depth", type=int, default=8,
                         help="How many levels deep to show. Trees in this project are trained "
                              "with max_depth=8, so 8 shows the FULL tree including leaf values. "
                              "Use a smaller number (e.g. 3) for a quicker, more zoomed-in view.")
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
    # Deeper trees have exponentially more leaf boxes on the bottom row, so a
    # fixed image size would squeeze everything unreadably small. Instead we
    # count how many leaves actually appear at the requested depth and scale
    # the figure (and font/dpi) to match.
    n_leaves = _count_leaves_at_depth(one_tree.tree_, args.max_depth)
    fig_width = max(16, n_leaves * 1.3)
    fig_height = max(8, (args.max_depth + 1) * 2.2)
    dpi = 150 if fig_width <= 40 else 100  # keep huge trees from becoming enormous files
    font_size = max(5, 9 - args.max_depth // 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(one_tree, feature_names=feature_names, filled=True,
              max_depth=args.max_depth, fontsize=font_size, ax=ax, rounded=True)
    png_path = f"{args.out_prefix}_{args.tree_index}.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Diagram saved to {png_path} ({n_leaves} leaf boxes at depth {args.max_depth}, "
          f"image size {fig_width:.0f}x{fig_height:.0f} inches)")


if __name__ == "__main__":
    main()
