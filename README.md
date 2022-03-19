# EVO-SPOTIS
This is Python 3 library for multi-criteria decision analysis with decision-maker preference identification.

# Installation
Downloading and installation of `evo_spotis` packege can be done using pip
```
pip install evo-spotis
```

# Methods
mcda_methods:
- SPOTIS (the Stable Preference Ordering Towards Ideal Solution method)

stochastic_algorithms:
- DE algorithm `DE_algorithm` (the Differential Evolution algorithm)

The DE algorithm is applied for the identification of criteria weights (decision-maker preferences) based on a training dataset with evaluated alternatives,
including alternatives performances (training features) and their ranking (target variable). The goal (fitness) function uses the correlation coefficient
of predicted ranking with real ranking. The predicted ranking is generated using the SPOTIS method and weights calculated by the DE algorithm in each DE iteration.
 It is a profit function. Therefore, higher values denote better results. Example of use of `evo_spotis` is included 
 in [examples](https://github.com/energyinpython/EVO-SPOTIS/tree/main/examples)

Other modules:
- `additions` including `rank_preference` method for ranking alternatives according to MCDA score
- `correlations` containing: 
	- Spearman rank correlation coefficient `spearman_coeff`, 
	- Weighted Spearman rank correlation coefficient `weigted_spearman_coeff`,
	- Pearson correlation coefficient `pearson_coeff`
- `normalizations` with methods for decision matrix normalization:
	- `linear_normalization` - Linear normalization,
	- `minmax_normalization` - Minimum- Maximum normalization,
	- `max_normalization` - Maximum normalization,
	- `sum_normalization` - Sum normalization,
	- `vector_normalization` - Vector normalization
- `weighting_methods` containing:
	- `entropy_weighting`.

