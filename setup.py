from setuptools import setup

setup(name='openai_b3',
      version='0.0.1',
      author='Robin Henry',
      author_email='robin@robinxhenry.com',
      packages=['openai_b3'],
      python_requires='>=3.8',
      install_requires=['gym_anm', 'tqdm', 'torch', 'stable_baselines3', 'tensorboard']
      )
