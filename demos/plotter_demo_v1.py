import numpy as np
from murakami_lab_modules.plotter import Plotter


def main():
    xs = np.linspace(0, 1, 100)
    y1 = (xs - 0.2) * (xs - 0.5) * (xs - 0.8) + np.random.random(xs.shape) * 0.03
    y2 = (xs + 0.2) * (xs - 0.6) * (xs - 1.8) + np.random.random(xs.shape) * 0.02
    y3 = (xs + 0.1) * (xs - 2.0) + np.random.random(xs.shape) * 0.04

    plotter = Plotter(
        fig_size=(10, 10),
        font_type='Times New Roman',
        n_data=3,
        change_shape=True
    )
    plotter.plot_and_scatter(x=xs, y=y1, label='y1')
    plotter.plot_and_scatter(x=xs, y=y2, label='y2')
    plotter.plot_and_scatter(x=xs, y=y3, label='y3')
    plotter.add_details(
        title='Plotter Demo. v1',
        x_label='x',
        y_label='y',
        x_lim=(0, 1),
        legend_outside=True
    )
    plotter.display()


if __name__ == '__main__':
    main()
