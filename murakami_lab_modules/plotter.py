import tkinter as tk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
            window_name: str = 'window',
            fig_size: tuple = (8.0, 6.0),
            main_font_size: float = 18.0,
            sub_font_size: float = 14.0,
            font_type: str = 'Times New Roman',
            main_line_width: int = 2,
            marker_size: float = 10.0,
            plot_line_width: int = 3,
            cmap_name: str = 'nipy_spectral',
            n_data: int = 5,
            change_color: bool = True,
            change_shape: bool = False,
            change_style: bool = False,
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
        self.n_data = n_data
        self.change_color = change_color
        self.change_shape = change_shape
        self.change_style = change_style
        self.background_color = background_color
        self.sub_background_color = sub_background_color
        self.line_darkness = line_darkness
        self.transparent_background = transparent_background

        if self.window_name:
            self.root = tk.Tk()
            self.root.title(self.window_name)
        else:
            self.root = None
        self._init_figure()

        self.color_idx = 0
        self.shape_idx = 0
        self.style_idx = 0
        self.z_order = 1

    def _init_figure(self):
        mpl.rcParams['font.family'] = self.font_type
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.linewidth'] = self.main_line_width
        self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        if self.root is not None:
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
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

    def _finish_plot(
            self,
            plot_type: str,
            change_color: bool,
            change_style: bool,
            change_shape: bool
    ):
        self.z_order += 1
        if self.change_color or change_color:
            self.color_idx = (self.color_idx + 1) % self.n_data
        if plot_type == 'plot' or plot_type == 'plot_and_scatter':
            if self.change_style or change_style:
                self.style_idx = (self.style_idx + 1) % len(self.style_list)
        elif plot_type == 'scatter' or plot_type == 'plot_and_scatter':
            if self.change_shape or change_shape:
                self.shape_idx = (self.shape_idx + 1) % len(self.shape_list)

    def _get_color(self, is_line: bool):
        if is_line:
            return [c * self.line_darkness for c in self.cmap(1 - (self.color_idx + 0.5) / self.n_data)[:3]]
        else:
            return [c for c in self.cmap(1 - (self.color_idx + 0.5) / self.n_data)[:3]]

    def plot(
            self,
            x: np.ndarray,
            y: np.ndarray,
            label: str = '',
            color: str | list = None,
            line_width: int = None,
            line_style: int | tuple | str = None,
            alpha: float = 0.8,
            change_color: bool = False,
            change_style: bool = False,
            **kwargs
    ):
        if color is None:
            color = self._get_color(is_line=True)
        if line_style is None:
            line_style = self.style_list[self.style_idx]
        elif type(line_style) is int:
            line_style = self.style_list[line_style]
        if line_width is None:
            line_width = self.plot_line_width

        self.ax.plot(
            x, y,
            label=label,
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            zorder=self.z_order,
            alpha=alpha,
            **kwargs
        )

        self._finish_plot(
            'plot_and_scatter',
            change_color=change_color,
            change_style=change_style,
            change_shape=False
        )

    def scatter(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_err: np.ndarray = None,
            y_err: np.ndarray = None,
            label: str = '',
            color: str | list = None,
            marker: int | str = None,
            marker_size: float = None,
            alpha: float = 0.8,
            change_color: bool = False,
            change_shape: bool = False,
            **kwargs
    ):
        if color is None:
            color = self._get_color(is_line=False)
        if marker is None:
            marker = self.shape_list[self.shape_idx]
        if marker_size is None:
            marker_size = self.marker_size
        elif type(marker) is int:
            marker = self.shape_list[marker]

        if x_err is not None:
            self.ax.errorbar(x, y, xerr=x_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1
        if y_err is not None:
            self.ax.errorbar(x, y, yerr=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1

        if alpha == 1.0:
            self.ax.scatter(
                x, y,
                s=marker_size ** 2,
                facecolor=color,
                edgecolors='k',
                linewidths=self.main_line_width,
                label=label,
                marker=marker,
                zorder=self.z_order,
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
                zorder=self.z_order,
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
                zorder=self.z_order,
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
                zorder=self.z_order,
                **kwargs
            )

        self._finish_plot(
            'plot_and_scatter',
            change_color=change_color,
            change_style=False,
            change_shape=change_shape
        )

    def plot_and_scatter(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_err: np.ndarray = None,
            y_err: np.ndarray = None,
            label: str = '',
            line_color: str | list = None,
            face_color: str | list = None,
            marker: int | str = None,
            marker_size: float = None,
            line_style: int | tuple | str = None,
            line_width: int = None,
            change_color: bool = False,
            change_style: bool = False,
            change_shape: bool = False,
            **kwargs
    ):
        if line_color is None:
            line_color = self._get_color(is_line=True)
        if face_color is None:
            face_color = self._get_color(is_line=False)
        if marker is None:
            marker = self.shape_list[self.shape_idx]
        if marker_size is None:
            marker_size = self.marker_size
        elif type(marker) is int:
            marker = self.shape_list[marker]
        if line_style is None:
            line_style = self.style_list[self.style_idx]
        elif type(line_style) is int:
            line_style = self.style_list[line_style]
        if line_width is None:
            line_width = self.plot_line_width

        if x_err is not None:
            self.ax.errorbar(x, y, x_err=x_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1
        if y_err is not None:
            self.ax.errorbar(x, y, yerr=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1

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
            zorder=self.z_order,
            **kwargs
        )
        self.z_order += 1

        self._finish_plot(
            'plot_and_scatter',
            change_color=change_color,
            change_style=change_style,
            change_shape=change_shape
        )

    def remove_plots(self, reset_idx: bool = True):
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()
        if reset_idx:
            self.color_idx = 0
            self.shape_idx = 0
            self.style_idx = 0
            self.z_order = 1

    def add_details(
            self,
            title: str = None,
            x_label: str = None,
            y_label: str = None,
            x_lim: tuple = None,
            y_lim: tuple = None,
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
        self.canvas.draw()
        self.root.update()

    def set_aspect(self, aspect: str | float, adjustable: str = None, anchor: str = None, share: bool = False):
        self.ax.set_aspect(aspect, adjustable=adjustable, anchor=anchor, share=share)

    def save_fig(self, name: str):
        self.fig.savefig(f'{name}.png', transparent=self.transparent_background)

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
    """
    matplotlib を用いた汎用ヒストグラム描画関数

    Parameters
    ----------
    data : array-like
        ヒストグラムを作成する1次元データ
    bins : int or sequence, default=30
        ビン数またはビン境界
    range : tuple, optional
        (min, max) の表示範囲
    density : bool, default=False
        Trueの場合、確率密度として正規化
    title : str, optional
        図のタイトル
    xlabel : str, optional
        x軸ラベル
    ylabel : str, optional
        y軸ラベル（未指定時は自動設定）
    log : bool, default=False
        Trueの場合、y軸を対数スケールにする
    save_path : str, optional
        指定したパスに図を保存（例: "hist.png"）
    show : bool, default=True
        Trueの場合、plt.show() を実行
    """

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
