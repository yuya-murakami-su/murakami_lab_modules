import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')

from murakami_lab_modules.visualization import Plotter


def test_plotter_has_research_colormap_aliases():
    names = Plotter.available_colormaps()

    assert 'blue_white_red' in names
    assert 'white_orange' in names
    assert Plotter.get_colormap('blue_white_red').N > 0


def test_plotter_draws_contourf_contour_and_removes_colorbar(tmp_path):
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 17)
    xx, yy = np.meshgrid(x, y)
    z = xx ** 2 - yy ** 2
    plotter = Plotter()

    filled = plotter.contourf(x=x, y=y, z=z, cmap='blue_white_red', colorbar_label='value')
    lines = plotter.contour(x=xx, y=yy, z=z, levels=5, label=True)
    plotter.add_details(x_label='x', y_label='y')
    plotter.save_fig(tmp_path / 'contour.png')

    assert filled is not None
    assert lines is not None
    assert len(plotter.colorbars) == 1
    assert (tmp_path / 'contour.png').exists()

    plotter.remove_plots()

    assert len(plotter.ax.collections) == 0
    assert plotter.colorbars == []
    plotter.close()


def test_plotter_validates_contour_shapes():
    plotter = Plotter()

    with pytest.raises(ValueError, match='z.shape must be'):
        plotter.contourf(x=np.arange(3), y=np.arange(4), z=np.zeros((3, 4)))

    plotter.close()
