import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        'Plotting utilities require matplotlib. '
        'Install it with `pip install murakami_lab_modules[plot]`.'
    ) from e


class Plotter:
    shape_list = ('o', 's', '^', 'D', 'v')
    style_list = (
        (0, (1, 0)),
        (0, (5, 1)),
        (0, (1, 1)),
        (0, (5, 1, 1, 1)),
        (0, (4, 2, 1, 1, 1, 2))
    )

    def __init__(
            self,
            window_name: str = '',
            fig_size: tuple[float, float] = (8.0, 6.0),
            main_font_size: float = 18.0,
            sub_font_size: float = 14.0,
            font_type: str = 'Times New Roman',
            main_line_width: int = 2,
            marker_size: float = 10.0,
            plot_line_width: int = 3,
            cmap_name: str = 'nipy_spectral',
            n_data: int = 5,
            cycle_color: bool = True,
            cycle_marker: bool = True,
            cycle_line_style: bool = False,
            background_color: str = '#ffffff',
            sub_background_color: str = '#f2f2f2',
            line_darkness: float = 0.8,
            transparent_background: bool = False
    ):
        self.window_name = window_name
        self.fig_size = fig_size
        self.main_font_size = main_font_size
        self.sub_font_size = sub_font_size
        self.font_type = font_type
        self.main_line_width = main_line_width

        self.marker_size = marker_size
        self.plot_line_width = plot_line_width
        self.cmap_name = cmap_name
        if n_data < 1:
            raise ValueError(f'n_data must be >= 1. {n_data} was given.')
        self.n_data = n_data
        self.cycle_color = cycle_color
        self.cycle_marker = cycle_marker
        self.cycle_line_style = cycle_line_style
        self.background_color = background_color
        self.sub_background_color = sub_background_color
        self.line_darkness = line_darkness
        self.transparent_background = transparent_background

        if self.window_name:
            try:
                import tkinter as tk
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            except ImportError as e:
                raise ImportError(
                    "Interactive Plotter windows require tkinter and matplotlib's TkAgg backend. "
                    "Use window_name='' for file-only plotting."
                ) from e
            self.root = tk.Tk()
            self.root.title(self.window_name)
            self._figure_canvas_cls = FigureCanvasTkAgg
        else:
            self.root = None
            self._figure_canvas_cls = None
        self._init_figure()

        self.series_idx = 0
        self.base_z_order = 1

    def _init_figure(self):
        plt.rcParams['font.family'] = self.font_type
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.linewidth'] = self.main_line_width
        self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        if self.root is not None:
            self.canvas = self._figure_canvas_cls(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack()

        self.ax.tick_params(
            labelsize=self.sub_font_size,
            top=True,
            right=True,
            direction='in',
            width=self.main_line_width
        )
        self.ax.set_facecolor(self.background_color)
        if self.transparent_background:
            self.fig.patch.set_alpha(0.0)
        else:
            self.fig.patch.set_alpha(1.0)
        self.ax.patch.set_alpha(1.0)
        self.cmap = plt.get_cmap(self.cmap_name)

    def _get_color(self, series: int, is_line: bool):
        color_idx = series % self.n_data if self.cycle_color else 0
        if is_line:
            return [c * self.line_darkness for c in self.cmap(1 - (color_idx + 0.5) / self.n_data)[:3]]
        else:
            return [c for c in self.cmap(1 - (color_idx + 0.5) / self.n_data)[:3]]

    def _get_marker(self, series: int):
        marker_idx = series % len(self.shape_list) if self.cycle_marker else 0
        return self.shape_list[marker_idx]

    def _get_line_style(self, series: int):
        style_idx = series % len(self.style_list) if self.cycle_line_style else 0
        return self.style_list[style_idx]

    def _get_series(self, series: int = None) -> tuple[int, bool]:
        if series is None:
            return self.series_idx, True
        if type(series) is not int:
            raise TypeError(f'series must be int or None. {type(series)} was given.')
        if series < 0:
            raise ValueError(f'series must be non-negative. {series} was given.')
        return series, False

    def _finish_series(self, advance_series: bool):
        if advance_series:
            self.series_idx += 1

    def plot(
            self,
            x: np.ndarray,
            y: np.ndarray,
            label: str = '',
            color: str | list[float] = None,
            line_width: int = None,
            line_style: int | tuple | str = None,
            alpha: float = 0.8,
            series: int = None,
            **kwargs
    ):
        series, advance_series = self._get_series(series)
        if color is None:
            color = self._get_color(series, is_line=True)
        if line_style is None:
            line_style = self._get_line_style(series)
        elif type(line_style) is int:
            line_style = self.style_list[line_style]
        if line_width is None:
            line_width = self.plot_line_width
        z_order = self.base_z_order + series

        self.ax.plot(
            x, y,
            label=label,
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            zorder=z_order,
            alpha=alpha,
            **kwargs
        )

        self._finish_series(advance_series)

    def scatter(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_err: np.ndarray = None,
            y_err: np.ndarray = None,
            label: str = '',
            color: str | list[float] = None,
            marker: int | str = None,
            marker_size: float = None,
            alpha: float = 0.8,
            series: int = None,
            **kwargs
    ):
        series, advance_series = self._get_series(series)
        if color is None:
            color = self._get_color(series, is_line=False)
        if marker is None:
            marker = self._get_marker(series)
        elif type(marker) is int:
            marker = self.shape_list[marker]
        if marker_size is None:
            marker_size = self.marker_size
        z_order = self.base_z_order + series

        if x_err is not None:
            self.ax.errorbar(x, y, xerr=x_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=z_order)
        if y_err is not None:
            self.ax.errorbar(x, y, yerr=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=z_order)

        if alpha == 1.0:
            self.ax.scatter(
                x, y,
                s=marker_size ** 2,
                facecolor=color,
                edgecolors='k',
                linewidths=self.main_line_width,
                label=label,
                marker=marker,
                zorder=z_order,
                **kwargs
            )
        else:
            self.ax.scatter(
                x, y,
                s=marker_size ** 2,
                facecolor=color,
                alpha=alpha,
                linewidths=0,
                marker=marker,
                zorder=z_order,
                **kwargs
            )
            self.ax.scatter(
                x, y,
                s=marker_size ** 2,
                facecolor='none',
                edgecolors='k',
                linewidths=self.main_line_width,
                alpha=1.0,
                marker=marker,
                zorder=z_order,
                **kwargs
            )
            self.ax.scatter(
                [], [],
                label=label,
                s=marker_size ** 2,
                facecolor=color,
                edgecolors='k',
                linewidths=self.main_line_width,
                marker=marker,
                zorder=z_order,
                **kwargs
            )

        self._finish_series(advance_series)

    def plot_and_scatter(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_err: np.ndarray = None,
            y_err: np.ndarray = None,
            label: str = '',
            line_color: str | list[float] = None,
            face_color: str | list[float] = None,
            marker: int | str = None,
            marker_size: float = None,
            line_style: int | tuple | str = None,
            line_width: int = None,
            series: int = None,
            **kwargs
    ):
        series, advance_series = self._get_series(series)
        if line_color is None:
            line_color = self._get_color(series, is_line=True)
        if face_color is None:
            face_color = self._get_color(series, is_line=False)
        if marker is None:
            marker = self._get_marker(series)
        elif type(marker) is int:
            marker = self.shape_list[marker]
        if marker_size is None:
            marker_size = self.marker_size
        if line_style is None:
            line_style = self._get_line_style(series)
        elif type(line_style) is int:
            line_style = self.style_list[line_style]
        if line_width is None:
            line_width = self.plot_line_width
        z_order = self.base_z_order + series

        if x_err is not None:
            self.ax.errorbar(x, y, xerr=x_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=z_order)
        if y_err is not None:
            self.ax.errorbar(x, y, yerr=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=z_order)

        self.ax.plot(
            x, y,
            label=label,
            marker=marker,
            markersize=marker_size,
            markerfacecolor=face_color,
            markeredgecolor='k',
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
            zorder=z_order,
            **kwargs
        )

        self._finish_series(advance_series)

    def remove_plots(self, reset_idx: bool = True):
        for line in list(self.ax.lines):
            line.remove()
        for collection in list(self.ax.collections):
            collection.remove()
        if reset_idx:
            self.series_idx = 0

    def add_details(
            self,
            title: str = None,
            x_label: str = None,
            y_label: str = None,
            x_lim: tuple[float, float] = None,
            y_lim: tuple[float, float] = None,
            x_log: bool = None,
            y_log: bool = None,
            legend_inside: bool = None,
            legend_outside: bool = None,
    ):
        if title is not None:
            self.ax.set_title(title, fontsize=self.main_font_size)
        if x_label is not None:
            self.ax.set_xlabel(x_label, fontsize=self.main_font_size)
        if y_label is not None:
            self.ax.set_ylabel(y_label, fontsize=self.main_font_size)
        if x_lim is not None:
            self.ax.set_xlim(*x_lim)
        if y_lim is not None:
            self.ax.set_ylim(*y_lim)
        if x_log is not None:
            if x_log:
                self.ax.set_xscale('log')
            else:
                self.ax.set_xscale('linear')
        if y_log is not None:
            if y_log:
                self.ax.set_yscale('log')
            else:
                self.ax.set_yscale('linear')
        if legend_inside is not None and legend_inside:
            legend = self.ax.get_legend()
            if legend:
                legend.remove()
            legend = self.ax.legend(
                fontsize=self.main_font_size,
                edgecolor='k',
                facecolor=self.sub_background_color
            )
            frame = legend.get_frame()
            frame.set_linewidth(self.main_line_width)
        if legend_outside is not None and legend_outside:
            legend = self.ax.get_legend()
            if legend:
                legend.remove()
            legend = self.ax.legend(
                fontsize=self.main_font_size,
                edgecolor='k',
                facecolor=self.sub_background_color,
                loc='upper left',
                bbox_to_anchor=(1, 1)
            )
            frame = legend.get_frame()
            frame.set_linewidth(self.main_line_width)
        self.fig.tight_layout()

    def update(self):
        if self.root is None:
            self.fig.canvas.draw_idle()
            return
        self.canvas.draw()
        self.root.update()

    def set_aspect(self, aspect: str | float, adjustable: str = None, anchor: str = None, share: bool = False):
        self.ax.set_aspect(aspect, adjustable=adjustable, anchor=anchor, share=share)

    def save_fig(self, save_path: str | Path):
        save_path = Path(save_path)
        if save_path.suffix == '':
            save_path = save_path.with_suffix('.png')
        self.fig.savefig(save_path, transparent=self.transparent_background)

    @staticmethod
    def display():
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.root is not None:
            self.root.destroy()
        plt.close(self.fig)


def plot_histogram(
    data,
    bins=30,
    range=None,
    density=False,
    title=None,
    xlabel=None,
    ylabel=None,
    log=False,
    save_path=None,
    show=True
):

    data = np.asarray(data)

    plt.figure()
    plt.hist(
        data,
        bins=bins,
        range=range,
        density=density
    )

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Density" if density else "Count")

    if log:
        plt.yscale("log")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv(r"C:\Users\YuyaMurakami\Desktop\Book1.csv", encoding='cp932')
    for name in data.columns:
        plot_histogram(data=data[name], show=True)
