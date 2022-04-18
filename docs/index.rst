Welcome to evo_spotis documentation!
===================================

This is Python 3 library for multi-criteria decision analysis with decision-maker preference identification.
This library includes:

- the SPOTIS method ``SPOTIS``
	
- Correlation coefficients:

	- ``spearman`` (Spearman rank correlation coefficient)
	- ``weighted_spearman`` (Weighted Spearman rank correlation coefficient)
	- ``pearson_coeff`` (Pearson correlation coefficient)
	
- Methods for normalization of decision matrix:

	- ``linear_normalization`` (Linear normalization)
	- ``minmax_normalization`` (Minimum-Maximum normalization)
	- ``max_normalization`` (Maximum normalization)
	- ``sum_normalization`` (Sum normalization)
	- ``vector_normalization`` (Vector normalization)
	
- Method for objective determination of criteria weights (weighting method)`entropy_weighting` (Entropy weighting method)
	
- additions:

	- ``rank_preferences`` (Method for ordering alternatives according to their preference values obtained with MCDA methods)
	
Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

	:maxdepth: 2

	usage
	example
	autoapi/index
