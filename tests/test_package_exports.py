def test_top_level_exports_common_api():
    from murakami_lab_modules import (
        DataHandler,
        DataFitting,
        FeedForwardNeuralNetwork,
        InputGenerator,
        ModelHandler,
        Optimizer,
        Regularization,
        relative_mse_loss,
    )

    assert DataHandler.__name__ == 'DataHandler'
    assert DataFitting.__name__ == 'DataFitting'
    assert FeedForwardNeuralNetwork.__name__ == 'FeedForwardNeuralNetwork'
    assert InputGenerator.__name__ == 'InputGenerator'
    assert ModelHandler.__name__ == 'ModelHandler'
    assert Optimizer.__name__ == 'Optimizer'
    assert Regularization.__name__ == 'Regularization'
    assert relative_mse_loss.__name__ == 'relative_mse_loss'
