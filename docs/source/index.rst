.. TabularBench documentation master file, created by
   sphinx-quickstart on Tue Aug 13 17:13:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TabularBench's documentation!
========================================
TabularBench is a comprehensive benchmark of robustness of tabular deep learning classification models. The benchmark implements 3 Tabular attacks: MOEVa, CAPGD and CAA.
And support 5 datasets, 5 tabular model architectures and 7 data augmentation mechanisms.

The benchmark provides pre-processed constrained datasets, as well as pre-trained robust tabular models.
The results of the benchmark can be found on [TabularBench website](https://serval-uni-lu.github.io/tabularbench/)

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   about
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   datasets
   constraints
   models
   attacks

.. toctree::
   :maxdepth: 2
   :caption: Contribute

   contribute

How to cite?
==================

.. code-block:: none

   @article{simonetto2024constrained,
     title={Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data},
     author={Simonetto, Thibault and Ghamizi, Salah and Cordy, Maxime},
     journal={arXiv preprint arXiv:2406.00775},
     year={2024}
   }

Search
==================

* :ref:`search`
