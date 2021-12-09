# PaddlePaddle Ascend910 Plugin Demo

This sample is a simple demo shows how to implement, build, install and run a PaddlePaddle plugin.

## Supported OS

- Linux

## Prerequisites

- CMake >= 3.10
- Git >= 1.8
- Python >= 3.6

## build & run 

1. Install the latest paddlepaddle

```bash
$ pip install paddlepaddle
```

2. In the `demo` code folder, configure and build

```bash
$ mkdir build
$ cd build
$ cmake .. -DWITH_KERNELS=ON
$ make
```

3. Install the plugin python wheel

```bash
$ pip install dist/*.whl
```

4. Now we can run the paddlepaddle and use the plug-in device

```python
$ python
>>> import paddle
>>> paddle.fluid.core.list_all_pluggable_device()
['Ascend910']
>>> paddle.set_device('Ascend910')
>>> x = paddle.to_tensor([1])
>>> x
Tensor(shape=[1], dtype=int64, place=PluggableDevicePlace(Ascend910: 0), stop_gradient=True,
       [1])
```
