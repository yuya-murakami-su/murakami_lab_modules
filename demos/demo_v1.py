import numpy as np
import pandas as pd
import torch.nn
from murakami_lab_modules.data_handler import DataHandler
from murakami_lab_modules.optimizer import Optimizer
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.model_handler import ModelHandler, DataFitting
from murakami_lab_modules.plotter import Plotter
from murakami_lab_modules.callbacks import LossMonitor
from murakami_lab_modules.predictor import NNPredictor


def main():
    # Define ground truth / 真の関数の定義
    ground_truth = lambda x: 1 - np.exp(- (2 * x[:, 0] + 0.3 * (x[:, 1] + 1) ** 2.6)) - x[:, 0] / 10
    x_ground_truth = [
        np.stack([np.ones(1000) * 0.5, np.linspace(0, 20, 1000)]).T,
        np.stack([np.ones(1000) * 1.0, np.linspace(0, 20, 1000)]).T,
        np.stack([np.ones(1000) * 2.0, np.linspace(0, 20, 1000)]).T
    ]
    y_ground_truth = [ground_truth(x_ground_truth[i]) for i in range(3)]

    # Define virtual data / 仮想データの定義
    np.random.seed(2025)  # Fix random seed
    n_data = 15
    x_data = [
        np.stack([np.ones(n_data) * 0.5, np.random.rand(n_data) * 5.0]).T,
        np.stack([np.ones(n_data) * 1.0, np.random.rand(n_data) * 5.0]).T,
        np.stack([np.ones(n_data) * 2.0, np.random.rand(n_data) * 5.0]).T
    ]
    noise_level = 0.01  # Add random noise
    y_data = [ground_truth(x_data[i]) * (1 + (np.random.rand(n_data) - 0.5) * noise_level) for i in range(3)]

    # Save dataset for machine learning / 機械学習用のデータセットとして保存
    pd.DataFrame(np.vstack(x_data), columns=('x1', 'x2')).to_csv('demo_data\\demo_v1_x.csv')
    pd.DataFrame(np.hstack(y_data)[:, None], columns=('y',)).to_csv('demo_data\\demo_v1_y.csv')

    # Define data handler for control datasets / データセット制御クラスの定義
    data_handler = DataHandler(
        input_data_path='demo_data\\demo_v1_x.csv',
        input_idx=['x1', 'x2'],
        output_data_path='demo_data\\demo_v1_y.csv',
        output_idx=['y'],
        batch_size=8,
        device_name='cpu',
        split_type='random_split',
        split_ratio=(0.9, 0.1, 0.0)  # (# of training data, # of validation data, # of test data)
    )

    # Define data fitting method during machine learning / データフィッティング手法の定義
    data_fitting = DataFitting(
        data_handler=data_handler,
        loss_criteria=torch.nn.MSELoss()
    )

    # Define optimizer / 最適化アルゴリズムの定義
    optimizer = Optimizer(
        algorithm=torch.optim.Adam,
        lr=1e-3
    )

    # Define neural network / ニューラルネットワークの定義
    neural_network = FeedForwardNeuralNetwork(
        n_input=2,
        n_output=1,
        n_layer=2,
        n_node=100,
        activation=torch.nn.Tanh(),
        random_seed=2025
    )

    # Define model handler / モデル制御用クラスの設定
    model_handler = ModelHandler(
        nn=neural_network,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=10_000,
        early_stop=300,
        callback_epoch=10,
        callbacks=(LossMonitor(need_data=True, need_reg=True),)
    )

    # Start training / 機械学習の実行
    model_handler()

    # Define neural network predictor / ニューラルネットワーク予測モデルの定義
    predictor = NNPredictor(
        model_path=model_handler.model_path,
        nn_class=FeedForwardNeuralNetwork
    )

    # Visualize result / 結果の可視化
    plotter = Plotter(n_data=3)
    plotter.plot(
        x=x_ground_truth[0][:, 1], y=y_ground_truth[0], label=r'Truth $(y=0.5)$', alpha=0.2, line_width=10
    )
    plotter.plot(
        x=x_ground_truth[1][:, 1], y=y_ground_truth[1], label=r'Truth $(y=1.0)$', alpha=0.2, line_width=10
    )
    plotter.plot(
        x=x_ground_truth[2][:, 1], y=y_ground_truth[2], label=r'Truth $(y=2.0)$', alpha=0.2, line_width=10
    )
    plotter.plot(x=x_ground_truth[0][:, 1], y=predictor(x_ground_truth[0]), label=r'NN $(y=0.5)$')
    plotter.plot(x=x_ground_truth[1][:, 1], y=predictor(x_ground_truth[1]), label=r'NN $(y=1.0)$')
    plotter.plot(x=x_ground_truth[2][:, 1], y=predictor(x_ground_truth[2]), label=r'NN $(y=2.0)$')
    plotter.scatter(x=x_data[0][:, 1], y=y_data[0], label=r'Data $(y=0.5)$', marker_size=4)
    plotter.scatter(x=x_data[1][:, 1], y=y_data[1], label=r'Data $(y=1.0)$', marker_size=4)
    plotter.scatter(x=x_data[2][:, 1], y=y_data[2], label=r'Data $(y=2.0)$', marker_size=4)
    plotter.add_details(x_label=r'$x$', y_label=r'$y$', x_lim=(0, 10), legend_outside=True)
    plotter.display()


if __name__ == '__main__':
    main()
