import os
from murakami_lab_modules.data_handler import DataHandler
from murakami_lab_modules.optimizer import OptimizerWithWarmup
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork
from murakami_lab_modules.model_handler import (
    ModelHandler, DataFitting, get_relative_error, get_mean_squared_error, get_absolute_error
)


def main():
    data_handler = DataHandler(
        input_data_path='demo_data\\test_data_1.csv',
        input_idx=['x', 'y', 'z'],
        output_idx=['f'],
        batch_size=128,
        device_name='cpu',
        split_type='random_split',
        split_ratio=(0.8, 0.1, 0.1)
    )
    optimizer = OptimizerWithWarmup(
        init_lr=1e-3,
        init_epoch=100,
        final_lr=1e-4,
        log_scale=True
    )
    neural_network = FeedForwardNeuralNetwork(
        n_input=3,
        n_output=1,
        n_layer=2,
        n_node=30
    )
    data_fitting = DataFitting(
        data_handler=data_handler,
        check_test=True,
        save_prediction_metrics=(get_mean_squared_error(), get_relative_error(epsilon=1e-3), get_absolute_error()),
        save_normalized_metrics=(get_mean_squared_error(), get_absolute_error()),
        save_loss_monitor=True,
        save_parity_plot=True
    )
    model_handler = ModelHandler(
        nn=neural_network,
        optimizer=optimizer,
        data_fitting=data_fitting,
        train_epochs=10_000,
        early_stop=100,
        save_path=os.path.basename(__file__)[:-3],
        model_name=os.path.basename(__file__)[:-3],
        train_record_path=f'{os.path.basename(__file__)[:-3]}_record',
        callbacks=('loss_monitor', 'loss_evolution_saver'),
        callback_epoch=100
    )
    model_handler()


if __name__ == '__main__':
    main()
