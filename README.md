# Wind Field Reconstruction with Adaptive Random Fourier Features

This directory contains the code necessary to replicate all the results from the article by Kiessling et al
on wind field reconstruction with adaptive random Fourier features. In short, the directory can be divided into four distinct parts:

* `framework` sets up a basic `windfield` framework for wind construction. It is similar to standard machine learning libraries.
the `framework` folder also contains tools for plotting results and loading wind data, located in the `Data` folder.
* `error_estimation` Contains some quality-of-life functions for making predictions an analysing said predictions using the framework.
* `models` contains a selection of interpolation / regression models for wind velocity prediction.
* `notebooks` contains a set of user-guides for working within the windfield framework.

## Recommended material

If you are new here, a good way of getting accquainted the code works is to follow the procedure below.

* Read `TUTORIAL-Windfield_models.ipynb` to learn about how to define and train models within the framework.
* If you are interested in the Neural network interpolation approach, read the `neural_network_demo_tronstad.ipynb` notebook by Magnus Tronstad.
* Read `TUTORIAL-Windfield_scoring.ipynb` to learn how to make and analyse predictions.
* Read the `RESEARCH_Fourier_Windfield_optimisation.ipynb` notebook to learn about hyper parameter optimisation,
* and run the code `multiproc_fourier_features_analysis.py` to see how the hyper parameter optimisation was done in the article.

There is more content here for the interested reader, but the above points should set you up to be able to discover the rest by yourself.
Please don't hesitate to ask any questions about the code or results.
