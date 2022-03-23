import numpy as np
import sys
import copy
import random

from ..additions import rank_preferences
from ..correlations import spearman_coeff
from ..mcda_methods.spotis import SPOTIS


class DE_algorithm():
    def __init__(self,
    var_min = sys.float_info.epsilon,
    var_max = 1.0,
    max_it = 200,
    n_pop = 60,
    beta_min = 0.2,
    beta_max = 0.8,
    p_CR = 0.4):

        """Create DE object with initialization of setting parametres of DE

        Parameters
        ----------
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
        """
        self.var_min = var_min
        self.var_max = var_max
        self.max_it = max_it
        self.n_pop = n_pop
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.p_CR = p_CR


    def __call__(self, X_train, y_train, types, bounds, verbose = True):
        """
        Determine criteria weights using DE algorithm with the goal (fitness) function using 
        SPOTIS method and Spearman rank coefficient

        Parameters
        ----------
            X_train : ndarray
                Decision matrix containing training dataset of alternatives and their performances corresponding to the criteria
            y_train: ndarray
                Ranking of training decision matrix which is the targer variable
            types : ndarray
                Criteria types. Profit criteria are represented by 1 and cost by -1.
            bounds : ndarray
                Bounds contain minimum and maximum values of each criterion. Minimum and maximum cannot be the same.
            verbose : bool
                For True `verbose` value, which is default, information about Best Fitness value in each iteration will be displayed
                and for False value, it will not

        Returns
        -------
            ndarray
                Values of best solution representing criteria weights
            ndarray
                Best values of fitness function in each iteration required for visualization of fitness function.
            ndarray
                Mean values of fitness function in each iteration required for visualization of fitness function.
        """
        self.var_size = np.shape(X_train)[1]
        self.verbose = verbose
        return DE_algorithm._de_algorithm(self, X_train, y_train, types, bounds)


    def fitness_function(self, matrix, weights, types, bounds, y_train):
        spotis = SPOTIS()
        pref = spotis(matrix, weights, types, bounds)
        rank = rank_preferences(pref, reverse = False)
        return spearman_coeff(rank, y_train)


    def _generate_population(self, X_train, y_train, types, bounds):

        # Initialize population with individuals
        class Empty_individual:
            Solution = None
            Fitness = None

        class Best_sol:
            Solution = None
            Fitness = -np.inf # fitness function is goal-maximizing function

        # Generate population
        BestSol = Best_sol()
        NewSol = Empty_individual()

        pop = [Empty_individual() for i in range(self.n_pop)]
        for i in range(self.n_pop):
            pop[i].Solution = np.random.uniform(self.var_min, self.var_max, self.var_size)
            
            # pop[i].Solution represent vector of weights
            pop[i].Solution = pop[i].Solution / np.sum(pop[i].Solution)
            pop[i].Fitness = self.fitness_function(X_train, pop[i].Solution, types, bounds, y_train)
            
            if (pop[i].Fitness >= BestSol.Fitness): # fitness function is goal-maximizing
                BestSol = copy.deepcopy(pop[i])

        return pop, BestSol, NewSol

    def _crossover(self, u, v, aj):
        u[aj] = v[aj]
        R = np.random.rand(len(u))
        u[R <= self.p_CR] = v[R <= self.p_CR]
        return u


    @staticmethod
    def _de_algorithm(self, X_train, y_train, types, bounds):

        # Generate population with individuals
        pop, BestSol, NewSol = self._generate_population(X_train, y_train, types, bounds)
        
        BestFitness = np.zeros(self.max_it)
        MeanFitness = np.zeros(self.max_it)
        # DE Main Loop
        for it in range(self.max_it):
            mean_fitness_sum = 0
            for i in range(self.n_pop):
                x = copy.deepcopy(pop[i].Solution)

                # Mutation
                v_pop = np.arange(self.n_pop)
                v_pop = np.delete(v_pop, i)
                A = random.sample(list(v_pop), 3)
                
                beta = np.random.uniform(self.beta_min, self.beta_max, self.var_size)
                # DE/rand/1 strategy
                # v = pop[A[0]].Solution+beta*(pop[A[1]].Solution-pop[A[2]].Solution)
                # DE/best/1/ strategy
                v = BestSol.Solution+beta*(pop[A[0]].Solution-pop[A[1]].Solution)
                v[v < self.var_min] = self.var_min
                v[v > self.var_max] = self.var_max

                # Crossover
                u = copy.deepcopy(x)
                aj = np.random.randint(0, self.var_size)
                u = self._crossover(u, v, aj)

                NewSol.Solution = copy.deepcopy(u)
                # NewSol.Solution represents vector of weights
                NewSol.Solution = NewSol.Solution / np.sum(NewSol.Solution)
                # Calculate fitness function for new solution (weights)
                NewSol.Fitness = self.fitness_function(X_train, NewSol.Solution, types, bounds, y_train)
                mean_fitness_sum += NewSol.Fitness

                # Selection
                # If fitness function of new solution `NewSol` (here fitness function is profit type) is better than
                # fitness function of actual individual `pop[i]` assign `NewSol` to actual individual `pop[i]`
                if NewSol.Fitness >= pop[i].Fitness: # fitness function is goal-maximizing
                    pop[i] = copy.deepcopy(NewSol)
                    
                    # If Fitness of new best solution `pop[i]` is better than global best solution `BestSol` assign new best solution to `BestSol`
                    if pop[i].Fitness >= BestSol.Fitness: # fitness function is goal-maximizing
                        BestSol = copy.deepcopy(pop[i])

            # Save the best and mean fitness value for iteration for visualization
            BestFitness[it] = copy.deepcopy(BestSol.Fitness)
            MeanFitness[it] = mean_fitness_sum / self.n_pop
            
            # Show Information about Best Fitness function value in actual Iteration
            if self.verbose:
                sys.stderr.write('\rIteration: %d/%d, Best Fitness = %f' % (it+1, self.max_it, BestFitness[it]))
                sys.stderr.flush()

        return BestSol.Solution, BestFitness, MeanFitness