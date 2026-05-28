import torch

from murakami_lab_modules import utils
from murakami_lab_modules.neural_network import FeedForwardNeuralNetwork


def test_feed_forward_neural_network_forward_shape():
    nn = FeedForwardNeuralNetwork(
        n_input=2,
        n_output=3,
        n_layer=1,
        n_node=4,
        activation=torch.nn.Tanh(),
        random_seed=1,
    )
    y = nn(torch.ones(5, 2))
    assert y.shape == (5, 3)


def test_feed_forward_neural_network_config_can_recreate_model():
    nn = FeedForwardNeuralNetwork(
        n_input=2,
        n_output=1,
        n_layer=1,
        n_node=4,
        activation=torch.nn.ReLU(),
        random_seed=1,
    )
    config = nn.config_dict()
    nn_class = utils.import_object(config['class'])
    recreated = nn_class(**utils.deserialize_params(config['params']))

    assert isinstance(recreated.activation, torch.nn.ReLU)
    assert recreated(torch.ones(2, 2)).shape == (2, 1)
