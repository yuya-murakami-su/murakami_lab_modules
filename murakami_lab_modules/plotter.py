import tkinter as tk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
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
            window_name: str = '',
            fig_size: tuple = (8.0, 6.0),
            main_font_size: float = 18.0,
            sub_font_size: float = 14.0,
            font_type: str = 'Times New Roman',
            main_line_width: int = 2,
            point_size: float = 10.0,
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

        self.point_size = point_size
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

    def _finish_plot(self, plot_type: str):
        self.z_order += 1
        if self.change_color:
            self.color_idx = (self.color_idx + 1) % self.n_data
        if plot_type == 'plot' or plot_type == 'plot_and_scatter':
            if self.style_idx:
                self.style_idx = (self.style_idx + 1) % len(self.style_list)
        elif plot_type == 'scatter' or plot_type == 'plot_and_scatter':
            if self.change_shape:
                self.shape_idx = (self.shape_idx + 1) % len(self.style_list)

    def _get_color(self, is_line: bool):
        if is_line:
            return [c * self.line_darkness for c in self.cmap(1 - (self.color_idx + 0.5) / self.n_data)[:3]]
        else:
            return [c for c in self.cmap(1 - (self.color_idx + 0.5) / self.n_data)[:3]]

    def plot(
            self,
            x: np.ndarray | torch.Tensor,
            y: np.ndarray | torch.Tensor,
            label: str = '',
            color: str | list = None,
            line_width: int = None,
            alpha: float = 0.8,
            **kwargs
    ):
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        if type(y) is torch.Tensor:
            y = y.detach().cpu().numpy()
        if color is None:
            color = self._get_color(is_line=True)
        if line_width is None:
            line_width = self.plot_line_width

        self.ax.plot(
            x, y,
            label=label,
            color=color,
            linewidth=line_width,
            zorder=self.z_order,
            alpha=alpha,
            **kwargs
        )
        self._finish_plot('plot')

    def scatter(
            self,
            x: np.ndarray | torch.Tensor,
            y: np.ndarray | torch.Tensor,
            x_err: np.ndarray | torch.Tensor = None,
            y_err: np.ndarray | torch.Tensor = None,
            label: str = '',
            color: str | list = None,
            alpha: float = 0.8,
            **kwargs
    ):
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        if type(y) is torch.Tensor:
            y = y.detach().cpu().numpy()
        if color is None:
            color = self._get_color(is_line=False)

        if x_err is not None:
            self.ax.errorbar(x, y, x_err=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1
        if y_err is not None:
            self.ax.errorbar(x, y, yerr=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1

        if alpha == 1.0:
            self.ax.scatter(
                x, y,
                s=self.point_size ** 2,
                facecolor=color,
                edgecolors='k',
                linewidths=self.main_line_width,
                label=label,
                marker=self.shape_list[self.shape_idx],
                zorder=self.z_order,
                **kwargs
            )
        else:
            self.ax.scatter(
                x, y,
                s=self.point_size ** 2,
                facecolor=color,
                alpha=alpha,
                linewidths=0,
                marker=self.shape_list[self.shape_idx],
                zorder=self.z_order,
                **kwargs
            )
            self.ax.scatter(
                x, y,
                s=self.point_size ** 2,
                facecolor='none',
                edgecolors='k',
                linewidths=self.main_line_width,
                alpha=1.0,
                marker=self.shape_list[self.shape_idx],
                zorder=self.z_order,
                **kwargs
            )
            self.ax.scatter(
                [], [],
                label=label,
                s=self.point_size ** 2,
                facecolor=color,
                edgecolors='k',
                linewidths=self.main_line_width,
                marker=self.shape_list[self.shape_idx],
                zorder=self.z_order,
                **kwargs
            )

        self._finish_plot('scatter')

    def plot_and_scatter(
            self,
            x: np.ndarray | torch.Tensor,
            y: np.ndarray | torch.Tensor,
            x_err: np.ndarray | torch.Tensor = None,
            y_err: np.ndarray | torch.Tensor = None,
            label: str = '',
            line_color: str | list = None,
            face_color: str | list = None,
            line_width: int = None,
            **kwargs
    ):
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        if type(y) is torch.Tensor:
            y = y.detach().cpu().numpy()
        if line_color is None:
            line_color = self._get_color(is_line=True)
        if face_color is None:
            face_color = self._get_color(is_line=False)
        if line_width is None:
            line_width = self.plot_line_width

        if x_err is not None:
            self.ax.errorbar(x, y, x_err=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1
        if y_err is not None:
            self.ax.errorbar(x, y, yerr=y_err, color='k', elinewidth=1, capsize=4, fmt='none', zorder=self.z_order)
            self.z_order += 1

        self.ax.plot(
            x, y,
            label=label,
            marker=self.shape_list[self.shape_idx],
            markersize=self.point_size,
            markerfacecolor=face_color,
            markeredgecolor='k',
            color=line_color,
            linewidth=line_width,
            linestyle=self.style_list[self.style_idx],
            zorder=self.z_order,
            **kwargs
        )
        self.z_order += 1

        self._finish_plot('plot_and_scatter')

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
        if x_log is not None and x_log:
            self.ax.set_xscale('log')
        if y_log is not None and y_log:
            self.ax.set_yscale('log')
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

    def save_fig(self, name: str):
        self.fig.savefig(f'{name}.png', transparent=False)

    def display(self):
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.root is not None:
            self.root.destroy()
