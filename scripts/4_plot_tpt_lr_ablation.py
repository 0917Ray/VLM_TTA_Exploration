"""Plot experiment 4: one-step TPT learning-rate ablation."""

import importlib


_plotter = importlib.import_module("3_plot_tpt_selection_ablation")
generate_plots = _plotter.generate_plots


if __name__ == "__main__":
    _plotter.main()
