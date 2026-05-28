import torch

from murakami_lab_modules import utils
from murakami_lab_modules.models import FeedForwardNeuralNetwork


def test_feed_forward_neural_network_forward_shape():
    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=3,
        n_hidden_layers=1,
        hidden_dim=4,
        activation=torch.nn.Tanh(),
        random_seed=1,
    )
    y = nn(torch.ones(5, 2))
    assert y.shape == (5, 3)


def test_feed_forward_neural_network_config_can_recreate_model():
    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_hidden_layers=1,
        hidden_dim=4,
        activation=torch.nn.ReLU(),
        random_seed=1,
    )
    config = nn.config_dict()
    nn_class = utils.import_object(config['class'])
    recreated = nn_class(**utils.deserialize_params(config['params']))

    assert isinstance(recreated.activation, torch.nn.ReLU)
    assert recreated(torch.ones(2, 2)).shape == (2, 1)


def test_feed_forward_neural_network_can_include_batch_norm_and_dropout():
    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_hidden_layers=2,
        hidden_dim=4,
        activation=torch.nn.ReLU(),
        dropout=0.2,
        batch_norm=True,
        random_seed=1,
    )

    modules = list(nn.nn)

    assert sum(isinstance(module, torch.nn.BatchNorm1d) for module in modules) == 2
    assert sum(isinstance(module, torch.nn.Dropout) for module in modules) == 2
    assert nn.batch_norm is True
    assert nn.dropout == 0.2


def test_hidden_activation_modules_are_not_shared_between_hidden_layers():
    nn = FeedForwardNeuralNetwork(
        input_dim=2,
        output_dim=1,
        n_hidden_layers=2,
        hidden_dim=4,
        activation=torch.nn.PReLU(),
        random_seed=1,
    )

    activations = [module for module in nn.nn if isinstance(module, torch.nn.PReLU)]

    assert len(activations) == 2
    assert activations[0] is not activations[1]


def test_invalid_dropout_raises_clear_error():
    try:
        FeedForwardNeuralNetwork(dropout=1.0)
    except ValueError as e:
        assert 'dropout' in str(e)
    else:
        raise AssertionError('dropout >= 1.0 should raise ValueError.')
