Usage
=====

.. _installation:

Installation
------------

To use evo_spotis, first install it using pip:

.. code-block:: python

	pip install evo_spotis

Importing methods from evo_spotis package
-------------------------------------

Import MCDA methods from module `mcda_methods`:

.. code-block:: python

	from evo_spotis.mcda_methods import SPOTIS

Import weighting methods from module `weighting_methods`:

.. code-block:: python

	from evo_spotis import weighting_methods as mcda_weights

Import normalization methods from module `normalizations`:

.. code-block:: python

	from evo_spotis import normalizations as norms

Import correlation coefficient from module `correlations`:

.. code-block:: python

	from evo_spotis import correlations as corrs

Import method for ranking alternatives according to prefernce values from module `additions`:

.. code-block:: python

	from evo_spotis.additions import rank_preferences



Usage examples
----------------------


The SPOTIS method
__________________

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	weights : ndarray
		Vector with criteria weights
	types : ndarray
		Vector with criteria types
		
Returns
	ndarray
		Vector with preference values of alternatives. Alternatives have to be ranked in ascending order according to preference values.

.. code-block:: python

	import numpy as np
	from evo_spotis.mcda_methods import SPOTIS

	import numpy as np
	from evo_spotis.mcda_methods import SPOTIS
	from evo_spotis.additions import rank_preferences

	# provide decision matrix in array numpy.darray
	matrix = np.array([[15000, 4.3, 99, 42, 737],
		[15290, 5.0, 116, 42, 892],
		[15350, 5.0, 114, 45, 952],
		[15490, 5.3, 123, 45, 1120]])

	# provide criteria weights in array numpy.darray. All weights must sum to 1.
	weights = np.array([0.2941, 0.2353, 0.2353, 0.0588, 0.1765])

	# provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
	types = np.array([-1, -1, -1, 1, 1])

	# Determine minimum bounds of performance values for each criterion in decision matrix
	bounds_min = np.array([14000, 3, 80, 35, 650])

	# Determine maximum bounds of performance values for each criterion in decision matrix
	bounds_max = np.array([16000, 8, 140, 60, 1300])

	# Stack minimum and maximum bounds vertically using vstack. You will get a matrix that has two rows and a number of columns equal to the number of criteria
	bounds = np.vstack((bounds_min, bounds_max))

	# Create the SPOTIS method object
	spotis = SPOTIS()

	# Calculate the SPOTIS preference values of alternatives
	pref = spotis(matrix, weights, types, bounds)

	# Generate ranking of alternatives by sorting alternatives ascendingly according to the SPOTIS algorithm (reverse = False means sorting in ascending order) according to preference values
	rank = rank_preferences(pref, reverse = False)

	print('Preference values: ', np.round(pref, 4))
	print('Ranking: ', rank)
	
Output

.. code-block:: console

	Preference values:  [0.478  0.5781 0.5557 0.5801]
	Ranking:  [1 3 2 4]
	
	
Stochastic algorithm
______________________

The Differential Evolution algorithm `DE_algorithm` for criteria weights prediction

Parameters
	var_min : float
		Lower bound of weights values
	var_max : float
		Upper bound of weights values
	max_it : int
		Maximum number of iterations
	n_pop : int
		Number of individuals in population
	beta_min : float
		Lower bound of range for random F parameter for mutation
	beta_max : float
		Upper bound of range for random F parameter for mutation
	p_CR : float
		Crossover probability
		
		
.. code-block:: python

	# Create object of the DE_algorithm
	de_algorithm = DE_algorithm()
	# run DE algorithm providing decision matrix with training dataset `X_train`, target variable of training dataset `y_train` (ranking), criteria types `types` and `bounds` for the SPOTIS method
	# de_algorithm returns `BestSolution` representing predicted criteria weights and the best `BestFitness` and mean `MeanFitness` goal (fitness) fucntion values
	BestSolution, BestFitness, MeanFitness = de_algorithm(X_train, y_train, types, bounds)


Correlation coefficents
__________________________

Spearman correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from evo_spotis import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `spearman` coefficient
	coeff = corrs.spearman(R, Q)
	print('Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Spearman coeff:  0.9

	
	
Weighted Spearman correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from evo_spotis import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `weighted_spearman` coefficient
	coeff = corrs.weighted_spearman(R, Q)
	print('Weighted Spearman coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Weighted Spearman coeff:  0.8833


	
Pearson correlation coefficient

Parameters
	R : ndarray
		First vector containing values
	Q : ndarray
		Second vector containing values
Returns
	float
		Value of correlation coefficient between two vectors

.. code-block:: python

	import numpy as np
	from evo_spotis import correlations as corrs

	# Provide two vectors with rankings obtained with different MCDA methods
	R = np.array([1, 2, 3, 4, 5])
	Q = np.array([1, 3, 2, 4, 5])

	# Calculate the correlation using `pearson_coeff` coefficient
	coeff = corrs.pearson_coeff(R, Q)
	print('Pearson coeff: ', np.round(coeff, 4))
	
Output

.. code-block:: console

	Pearson coeff:  0.9
	
	
	
Method for criteria weights determination
___________________________________________

Entropy weighting method

Parameters
	matrix : ndarray
		Decision matrix with performance values of m alternatives and n criteria
Returns
	ndarray
		vector of criteria weights
		
.. code-block:: python

	import numpy as np
	from evo_spotis import weighting_methods as mcda_weights

	matrix = np.array([[30, 30, 38, 29],
	[19, 54, 86, 29],
	[19, 15, 85, 28.9],
	[68, 70, 60, 29]])
	
	weights = mcda_weights.entropy_weighting(matrix)
	
	print('Entropy weights: ', np.round(weights, 4))
	
Output

.. code-block:: console

	Entropy weights:  [0.463  0.3992 0.1378 0.    ]
	
	
Normalization methods
______________________

Here is an example of vector normalization usage. Other normalizations provided in module `normalizations`, namely `minmax_normalization`, `max_normalization`,
`sum_normalization`, `linear_normalization`, `multimoora_normalization` are used in analogous way.


Vector normalization

Parameters
	matrix : ndarray
		Decision matrix with m alternatives in rows and n criteria in columns
	types : ndarray
		Criteria types. Profit criteria are represented by 1 and cost by -1.
Returns
	ndarray
		Normalized decision matrix

.. code-block:: python

	matrix = np.array([[8, 7, 2, 1],
    [5, 3, 7, 5],
    [7, 5, 6, 4],
    [9, 9, 7, 3],
    [11, 10, 3, 7],
    [6, 9, 5, 4]])

    types = np.array([1, 1, 1, 1])

    norm_matrix = norms.vector_normalization(matrix, types)
    print('Normalized matrix: ', np.round(norm_matrix, 4))
	
Output

.. code-block:: console

	Normalized matrix:  [[0.4126 0.3769 0.1525 0.0928]
	 [0.2579 0.1615 0.5337 0.4642]
	 [0.361  0.2692 0.4575 0.3714]
	 [0.4641 0.4845 0.5337 0.2785]
	 [0.5673 0.5384 0.2287 0.6499]
	 [0.3094 0.4845 0.3812 0.3714]]

