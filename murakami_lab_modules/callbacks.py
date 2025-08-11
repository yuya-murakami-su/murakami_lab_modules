import os
import torch
import pandas as pd
from murakami_lab_modules.plotter import Plotter


class Callback:
    def on_train_begin(self, model_handler):
        pass

    def on_epoch_begin(self, model_handler):
        pass

    def on_call(self, model_handler):
        pass

    def on_epoch_end(self, model_handler):
        pass

    def on_train_end(self, model_handler):
        pass


class SaveLossMonitor(Callback):
    def on_train_end(self, model_handler):
        n_data, get_xy = model_handler.get_loss_info_fnc()
        plotter = Plotter(
            window_name='',
            n_data=n_data
        )
        x, ys, labels = get_xy(model_handler.evolution, model_handler.epoch)
        for y, label in zip(ys, labels):
            plotter.plot(x=x, y=y, label=label)

        plotter.add_details(
            title='Loss monitor',
            x_label='Training epochs [-]',
            y_label='Loss [-]',
            y_log=True,
            x_lim=(0, model_handler.epoch),
            legend_outside=True
        )
        plotter.save_fig(name=f'{model_handler.model_path}\\loss_monitor')
        plotter.close()


class SavePredictionResults(Callback):
    def on_train_begin(self, model_handler):
        if not model_handler.has_data:
            raise ValueError('SavePredictionResults callback cannot be used if the model does not have data_fitting.')

    def on_train_end(self, model_handler):
        with torch.no_grad():
            prediction_results = []
            for key in ['train', 'valid', 'test']:
                for x, y, label in model_handler.data_fitting.data_handler(key):
                    y_pred = model_handler.nn(x)

                    x_ = model_handler.data_fitting.data_handler.undo_normalize_x(x)
                    y_ = model_handler.data_fitting.data_handler.undo_normalize_y(y)
                    y_pred_ = model_handler.data_fitting.data_handler.undo_normalize_y(y_pred)

                    if (
                            model_handler.data_fitting.save_prediction_metrics or
                            model_handler.data_fitting.save_normalized_metrics
                    ):
                        evaluated = torch.hstack(
                            [metric(y_, y_pred_) for metric in model_handler.data_fitting.save_prediction_metrics] +
                            [metric(y, y_pred) for metric in model_handler.data_fitting.save_normalized_metrics]
                        )
                        prediction_results.append(torch.hstack([label, x_, y_, y_pred_, evaluated]))
                    else:
                        prediction_results.append(torch.hstack([label, x_, y_, y_pred_]))
            prediction_results = torch.vstack(prediction_results).cpu().numpy()

        columns = (
                ['label'] +
                [f'x_{i}' for i in range(model_handler.nn.n_input)] +
                [f'y_true_{i}' for i in range(model_handler.nn.n_output)] +
                [f'y_pred_{i}' for i in range(model_handler.nn.n_output)] +
                [f'{metric.__name__}_pred' for metric in model_handler.data_fitting.save_prediction_metrics] +
                [f'{metric.__name__}_norm' for metric in model_handler.data_fitting.save_normalized_metrics]
        )

        pd.DataFrame(prediction_results, columns=columns).to_csv(f'{model_handler.model_path}\\prediction_results.csv')


class SaveParityPlot(Callback):
    def on_train_begin(self, model_handler):
        if not model_handler.has_data:
            raise ValueError('SavePredictionResults callback cannot be used if the model does not have data_fitting.')

    def on_train_end(self, model_handler):
        y_max = torch.full([1, model_handler.nn.n_output], -torch.inf)
        y_min = torch.full([1, model_handler.nn.n_output], torch.inf)
        with torch.no_grad():
            results = {}
            for key in ['train', 'valid', 'test']:
                if model_handler.data_fitting.data_handler.n_data[key] == 0:
                    continue
                y_list, y_pred_list = [], []
                for x, y, label in model_handler.data_fitting.data_handler(key):
                    y_pred = model_handler.nn(x)
                    y_ = model_handler.data_fitting.data_handler.undo_normalize_y(y)
                    y_pred_ = model_handler.data_fitting.data_handler.undo_normalize_y(y_pred)
                    y_list.append(y_)
                    y_pred_list.append(y_pred_)

                    y_min, _ = torch.min(torch.vstack([y_, y_pred_, y_min]), dim=0, keepdim=True)
                    y_max, _ = torch.max(torch.vstack([y_, y_pred_, y_max]), dim=0, keepdim=True)

                results[key] = [torch.vstack(y_list), torch.vstack(y_pred_list)]

        os.makedirs(f'{model_handler.model_path}\\parity_plot')
        for y_idx in range(model_handler.nn.n_output):
            y_max_, y_min_ = y_max[0, y_idx], y_min[0, y_idx]
            dy = (y_max_ - y_min_) * 0.1
            total_plotter = Plotter(
                window_name='',
                n_data=3
            )
            total_plotter.plot(x=[y_min_ - dy, y_max_ + dy], y=[y_min_ - dy, y_max_ + dy], color='k', line_width=2)
            total_plotter.add_details(
                title=f'Parity plot ({y_idx=})',
                x_label=r'$y_{true}$',
                y_label=r'$y_{calc}$',
                x_lim=(y_min_ - dy, y_max_ + dy),
                y_lim=(y_min_ - dy, y_max_ + dy)
            )
            individual_plotter = Plotter(
                window_name='',
                n_data=3
            )
            individual_plotter.add_details(
                x_label=r'$y_{true}$',
                y_label=r'$y_{calc}$',
                x_lim=(y_min_ - dy, y_max_ + dy),
                y_lim=(y_min_ - dy, y_max_ + dy)
            )
            for key in ['train', 'valid', 'test']:
                if model_handler.data_fitting.data_handler.n_data[key] == 0:
                    continue
                total_plotter.scatter(x=results[key][0][:, y_idx], y=results[key][1][:, y_idx], label=key)
                individual_plotter.plot(
                    x=[y_min_ - dy, y_max_ + dy], y=[y_min_ - dy, y_max_ + dy], color='k', line_width=2
                )
                individual_plotter.scatter(x=results[key][0][:, y_idx], y=results[key][1][:, y_idx], label=key)
                individual_plotter.add_details(title=f'Parity plot ({y_idx=}, {key})')
                individual_plotter.save_fig(f'{model_handler.model_path}\\parity_plot\\parity_plot_y{y_idx}_{key}')
                individual_plotter.remove_plots(reset_idx=False)
            total_plotter.add_details(legend_inside=True)
            total_plotter.save_fig(f'{model_handler.model_path}\\parity_plot\\parity_plot_y{y_idx}')
            individual_plotter.close()
            total_plotter.close()


class LossMonitor(Callback):
    def __init__(self, need_data: bool = True, need_reg: bool = True):
        self.need_data = need_data
        self.need_reg = need_reg
        self.n_data = None
        self.get_xy = None
        self.plotter = None

    def on_train_begin(self, model_handler):
        self.n_data, self.get_xy = model_handler.get_loss_info_fnc(need_data=self.need_data, need_reg=self.need_reg)
        self.plotter = Plotter(
            window_name='loss_monitor',
            n_data=self.n_data
        )
        self.plotter.add_details(
            title='Loss monitor',
            x_label='Training epochs [-]',
            y_label='Loss [-]',
            y_log=True
        )

    def on_call(self, model_handler):
        self.plotter.remove_plots()
        x, ys, labels = self.get_xy(model_handler.evolution, model_handler.epoch)
        for y, label in zip(ys, labels):
            self.plotter.plot(x=x, y=y, label=label)
        self.plotter.add_details(x_lim=(0, model_handler.epoch), legend_outside=True)
        self.plotter.update()

    def on_train_end(self, model_handler):
        self.plotter.close()


class LossEvolutionSaver(Callback):
    def __init__(self, need_data: bool = True, need_reg: bool = True):
        self.need_data = need_data
        self.need_reg = need_reg
        self.n_data = None
        self.get_xy = None
        self.plotter = None

    def on_train_begin(self, model_handler):
        self.n_data, self.get_xy = model_handler.get_loss_info_fnc(need_data=self.need_data, need_reg=self.need_reg)
        os.makedirs(f'{model_handler.model_path}\\loss_evolution')
        self.plotter = Plotter(
            window_name='',
            n_data=self.n_data
        )
        self.plotter.add_details(
            title='Loss evolution',
            x_label='Training epochs [-]',
            y_label='Loss [-]',
            y_log=True
        )

    def on_call(self, model_handler):
        self.plotter.remove_plots()
        x, ys, labels = self.get_xy(model_handler.evolution, model_handler.epoch)
        for y, label in zip(ys, labels):
            self.plotter.plot(x=x, y=y, label=label)

        self.plotter.add_details(
            x_lim=(0, model_handler.epoch),
            legend_outside=True
        )
        self.plotter.save_fig(name=f'{model_handler.model_path}\\loss_evolution\\{model_handler.epoch + 1:0>6}')

    def on_train_end(self, model_handler):
        self.plotter.close()


class StateDictsSaver(Callback):
    def on_train_begin(self, model_handler):
        os.makedirs(f'{model_handler.model_path}\\state_dicts')

    def on_call(self, model_handler):
        torch.save(model_handler.state_dicts, f'{model_handler.model_path}\\state_dicts\\{model_handler.epoch + 1:0>6}')
