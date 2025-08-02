import torch
from . import utils


class InputGenerator:
    def __init__(
            self,
            size_of_generated_inputs: int,
            device_name: str,
            shuffle: bool = True,
            distribution: str = 'random',
            random_seed: int = 2025,
            **kwargs
    ):
        self.locals = utils.get_local_dict(locals())
        utils.initialize_random_seed(random_seed)

        self.size_of_generated_inputs = size_of_generated_inputs
        self.device_name = device_name
        self.device = utils.get_device(device_name)
        self.shuffle = shuffle
        self.distribution = distribution
        self.random_seed = random_seed
        self.kwargs = kwargs

        if distribution == 'random':
            if shuffle:
                self.generator = self._random_distribution(input_range=kwargs['input_range'])
            else:
                self.generator = self._fixed_random_distribution(input_range=kwargs['input_range'])
        elif distribution == 'fixed':
            if 'inputs' not in kwargs.keys():
                raise ValueError('inputs must be given when fixed distribution is chosen.')
            if not len(kwargs['inputs']) == size_of_generated_inputs:
                raise ValueError('length of inputs must be same as size_of_generated_inputs')
            self.generator = self._fixed_distribution(inputs=kwargs['inputs'])
        else:
            exec(f'self.generator = self._{distribution}(**kwargs)')

    def _random_distribution(self, input_range: tuple = None, **_: dict):
        if input_range is None:
            raise ValueError('input_range must be given when random distribution is chosen.')
        for range_ in input_range:
            if not len(range_) == 2:
                raise ValueError('length of tuples in input_range must be 2.')

        l_input = len(input_range)
        d_range = torch.tensor(
            [r[1] - r[0] for r in input_range],
            dtype=torch.float32,
            device=self.device
        ).view(1, -1)
        min_range = torch.tensor(
            [r[0] for r in input_range],
            dtype=torch.float32,
            device=self.device
        ).view(1, -1)

        def get_distribution():
            distribution = torch.rand(
                [self.size_of_generated_inputs, l_input],
                dtype=torch.float32,
                device=self.device
            ) * d_range + min_range
            return distribution.requires_grad_(True)

        return get_distribution

    def _fixed_random_distribution(self, input_range: tuple = None, **_: dict):
        if input_range is None:
            raise ValueError('input_range must be given when random distribution is chosen.')
        for range_ in input_range:
            if not len(range_) == 2:
                raise ValueError('length of tuples in input_range must be 2.')

        distribution = self._random_distribution(input_range)().detach()

        def get_distribution():
            return distribution.clone().requires_grad_(True)

        return get_distribution

    def _fixed_distribution(self, inputs: tuple = None, **_: dict):
        if inputs is None:
            raise ValueError('inputs must be given when random distribution is chosen.')
        distribution = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        def get_distribution():
            return distribution.clone().requires_grad_(True)

        return get_distribution

    def __call__(self):
        return self.generator()