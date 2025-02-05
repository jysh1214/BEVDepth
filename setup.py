import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension)


with open('README.md', 'r') as fh:
    long_description = fh.read()


def make_ext(
    name,
    module,
    sources,
    extra_args=[],
    extra_include_path=[],
):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}
    extension = CppExtension
    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name='BEVDepth',
    version='0.0.1',
    author='Megvii',
    author_email='liyinhao@megvii.com',
    description='Code for BEVDepth',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=None,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    ext_modules=[
        make_ext(
            name='voxel_pooling_train_ext',
            module='bevdepth.ops.voxel_pooling_train',
            sources=['src/voxel_pooling_train_forward.cpp'],
        ),
        make_ext(
            name='voxel_pooling_inference_ext',
            module='bevdepth.ops.voxel_pooling_inference',
            sources=['src/voxel_pooling_inference_forward.cpp'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
