from murakami_lab_modules.data_handler import DataHandler
from murakami_lab_modules.input_generator import InputGenerator
from murakami_lab_modules.optimizer import OptimizerWithWarmup
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork, AbstractNeuralNetwork
from murakami_lab_modules.model_handler import (
    ModelHandler, DataFitting, Regularization, get_relative_error, get_mean_squared_error, get_absolute_error
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
    input_generator = InputGenerator(
        size_of_generated_inputs=100,
        device_name=data_handler.device_name,
        shuffle=True,
        distribution='random',
        input_range=((0, 2), (1, 2), (-2, 3))
    )
    my_regularization = MyRegularization(
        input_generators=[input_generator],
        reg_func_name='regularization',
        reg_weights=[1.0, 1.0, 1.0]
    )
    optimizer = OptimizerWithWarmup(
        init_lr=1e-3,
        init_epoch=5000,
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
        regularization=my_regularization,
        train_epochs=10000,
        early_stop=10000,
        save_path=os.path.basename(__file__)[:-3],
        model_name=f'{os.path.basename(__file__)[:-3]}',
        train_record_path=f'{os.path.basename(__file__)[:-3]}_record',
        callbacks=('loss_monitor',),
        callback_epoch=100
    )
    model_handler()


class MyRegularization(Regularization):
    def regularization(self, data_handler: DataHandler, nn: AbstractNeuralNetwork):
        inputs = self.input_generators[0]()
        f = nn(inputs)
        d1_f_d_inputs = self.grad(y=f, x=inputs)
        d1_f_d_x, d1_f_d_y, d1_f_d_z = d1_f_d_inputs[:, 0], d1_f_d_inputs[:, 1], d1_f_d_inputs[:, 2]
        d2_f_d_x = self.grad(y=d1_f_d_inputs, x=inputs)[:, 0]
        return [
            d2_f_d_x.clamp_max_(max=0),
            d1_f_d_y.clamp_max_(max=0),
            d1_f_d_z.clamp_max_(max=0),
        ]


if __name__ == '__main__':
    main()
