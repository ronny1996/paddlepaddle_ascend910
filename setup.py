from setuptools import setup, Distribution, Extension
from setuptools.command.build_ext import build_ext
import os
import shutil

packages = []
package_data = {}


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

for pkg_dir in ['build/python/paddle-plugins/']:
    if os.path.exists(pkg_dir):
        shutil.rmtree(pkg_dir)
    os.makedirs(pkg_dir)

ext_modules = [Extension(name='paddle-plugins.libpaddle_ascend910',
                         sources=['runtime/runtime.cc'],
                         include_dirs=['/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/',
                                       '/opt/conda/lib/python3.7/site-packages/paddle/include/'],
                         library_dirs=[
                             '/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/'],
                         libraries=['ascendcl'])]

setup(
    name='paddle-ascend910',
    version='0.0.1',
    description='Paddle ascend910 plugin',
    long_description='',
    long_description_content_type="text/markdown",
    author_email="Paddle-better@baidu.com",
    maintainer="PaddlePaddle",
    maintainer_email="Paddle-better@baidu.com",
    project_urls={},
    license='Apache Software License',
    ext_modules=ext_modules,
    packages=[
        'paddle-plugins',
    ],
    include_package_data=True,
    package_data={
        '': ['*.so', '*.h', '*.py', '*.hpp'],
    },
    package_dir={
        '': 'build/python',
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    entry_points={
        'console_scripts': [
        ]
    },
    classifiers=[
    ],
    keywords='paddle ascend910 plugin',
)
