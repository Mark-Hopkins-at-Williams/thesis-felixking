from setuptools import setup, find_packages

setup(name='finetuning',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'torch',
          'transformers',
          'numpy',
          'matplotlib',
          'pandas',
          'evaluate',
          'tqdm'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      )

