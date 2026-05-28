# murakami_lab_modules

murakami_lab_modules is the library mainly for machine learning.
For the detailed instruction, please contact to the author.
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## How to install
```markdown
pip install git+https://github.com/yuya-murakami-su/murakami_lab_modules.git
```

## PyTorch dependency

**Important:** PyTorch is required by the core modules, but this package does not install it by default.
Install PyTorch first so you can choose the correct CPU or CUDA build for your environment.

For example, install the PyTorch build recommended for your system from the official PyTorch installation guide,
then install this package:

```markdown
pip install git+https://github.com/yuya-murakami-su/murakami_lab_modules.git
```

If you do not need a specific CPU/CUDA wheel, a convenience extra is available:

```markdown
pip install "git+https://github.com/yuya-murakami-su/murakami_lab_modules.git#egg=murakami_lab_modules[torch]"
```

This extra asks pip to install `torch`, but it does not guarantee a specific CPU or CUDA build.

## Optional dependencies

Plotting utilities:

```markdown
pip install "git+https://github.com/yuya-murakami-su/murakami_lab_modules.git#egg=murakami_lab_modules[plot]"
```

PCA/statistics utilities:

```markdown
pip install "git+https://github.com/yuya-murakami-su/murakami_lab_modules.git#egg=murakami_lab_modules[statistics]"
```

All optional dependencies:

```markdown
pip install "git+https://github.com/yuya-murakami-su/murakami_lab_modules.git#egg=murakami_lab_modules[all]"
```

## Author
```markdown
- Yuya Murakami  
- Shizuoka University, Japan
- July 21st, 2025  
- murakami.yuhya@cii.shizuoka.ac.jp
```
