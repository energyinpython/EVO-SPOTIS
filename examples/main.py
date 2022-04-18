import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evo_spotis.mcda_methods import SPOTIS
from evo_spotis.stochastic_algorithms import DE_algorithm
from evo_spotis.additions import rank_preferences
from evo_spotis import correlations as corrs
from evo_spotis import weighting_methods as mcda_weights

from objective_weighting import weighting_methods as pyweights

# Functions for result visualizations
def plot_scatter(data, model_compare):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        data: dataframe
        model_compare : list[list]

    Examples
    ----------
    >>> plot_scatter(data. model_compare)
    """
    sns.set_style("darkgrid")
    list_rank = np.arange(1, len(data) + 2, 4)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 12)
        ax.set_ylabel(el[1], fontsize = 12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        x_ticks = ax.xaxis.get_major_ticks()
        y_ticks = ax.yaxis.get_major_ticks()

        ax.set_xlim(-1, len(data) + 1)
        ax.set_ylim(0, len(data) + 1)

        ax.grid(True, linestyle = '--')
        ax.set_axisbelow(True)
    
        plt.tight_layout()
        plt.savefig('results/scatter_' + el[0] + '.pdf')
        plt.show()


def plot_fitness(BestFitness, MeanFitness):
    """
    Display line plot of best and mean fitness values in each DE iteration.

    Parameters
    ----------
        BestFitness : ndarray
            array with best fitness values for each DE iteration.
        MeanFitness : ndarray
            array with mean fitness values for each DE iteration.

    Examples
    ----------
    >>> plot_fitness(BestFitness, MeanFitness)
    """
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    ax.plot(BestFitness, label = 'Best fitness value')
    ax.plot(MeanFitness, label = 'Mean fitness value')
    ax.set_xlabel('Iterations', fontsize = 12)
    ax.set_ylabel('Fitness value', fontsize = 12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig('results/fitness.pdf')
    plt.show()


def plot_rankings(results):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        results : dataframe
            Dataframe with columns containing real and predicted rankings.

    Examples
    ---------
    >>> plot_rankings(results)
    """
    model_compare = []
    names = list(results.columns)
    model_compare = [[names[0], names[1]]]
    results = results.sort_values('Real rank')
    sns.set_style("darkgrid")
    plot_scatter(data = results, model_compare = model_compare)


def plot_weights(weights):
    """
    Display scatter plot comparing real and predicted weights

    Parameters
    -----------
        weights : dataframe
            Dataframe with columns containing real and predicted weights.

    Examples
    ----------
    >>> plot_weights(weights)
    """
    sns.set_style("darkgrid")
    step = 1
    list_rank = np.arange(1, len(weights) + 1, step)
    
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.scatter(x = list_rank, y = weights['Real weights'].to_numpy(), label = 'Real weights')
    ax.scatter(x = list_rank, y = weights['DE weights'].to_numpy(), label = 'DE weights')
    
    ax.set_xlabel('Criteria', fontsize = 12)
    ax.set_ylabel('Weight value', fontsize = 12)
    ax.set_xticks(list_rank)

    ax.set_xticklabels(list(weights.index))
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    plt.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('results/weights_comparison.pdf')
    plt.show()

# Create dictionary class helpful in saving results of correlation coefficient values between compared rankings
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


def main():
    # Load dataset from CSV file to dataframe
    # Rows contain alternatives and columns contain criteria. The last row contain criteria types and the last column contain reference ranking of alternatives
    filename = 'mobile_phones2000.csv'
    data = pd.read_csv(filename)
    # In this example criteria types are given in the last row of CSV file so load criteria types from the last row
    types = data.iloc[len(data) - 1, :].to_numpy()
    # Load examplary dataset with the first 200 alternatives
    df_data = data.iloc[:200, :]
    # Convert dataframe to numpy array (ndarray) for faster calculations
    whole_matrix = df_data.to_numpy()

    # Determine bounds of alternatives performances for the SPOTIS method
    bounds_min = np.amin(whole_matrix, axis = 0)
    bounds_max = np.amax(whole_matrix, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))

    # Load train and test datasets
    # y_train which is training target variable with the reference ranking of alternatives is given in the last column
    train_df = pd.read_csv('train.csv', index_col = 'Ai')
    X_train = train_df.iloc[:len(train_df) - 1, :-1].to_numpy()
    y_train = train_df.iloc[:len(train_df) - 1, -1].to_numpy()

    # y_test which is test target variable with the reference ranking of alternatives is given in the last column
    test_df = pd.read_csv('test.csv', index_col = 'Ai')
    X_test = test_df.iloc[:len(test_df) - 1, :-1].to_numpy()
    y_test = test_df.iloc[:len(test_df) - 1, -1].to_numpy()

    # In this example real weights are determined using Entropy weighting method
    train_weights = mcda_weights.entropy_weighting(X_train)
    # Load symbols of criteria in columns 
    cols = [r'$C_{' + str(y) + '}$' for y in range(1, data.shape[1] + 1)]


    
    # Run the DE algorithm
    # BestSolution represents weights generated by the DE algorithm
    # BestFitness and MeanFitness are DE fitness function values
    de_algorithm = DE_algorithm()
    BestSolution, BestFitness, MeanFitness = de_algorithm(X_train, y_train, types, bounds)

    # Save results in dataframes
    # Weights
    weights = pd.DataFrame(index = cols)
    weights['Real weights'] = train_weights
    weights['DE weights'] = BestSolution
    weights = weights.rename_axis('Cj')
    weights.to_csv('results/best_weights_de.csv')

    # Calculate correlation between real weights and predicted weights of train alternatives using Pearson correlation coefficient
    print('\nWeights correlation: ', corrs.pearson_coeff(train_weights, BestSolution))
    plot_weights(weights)

    # Save the best and mean Fitness values in datafarme and plot chart of Fitness values
    fitness_best = pd.DataFrame()
    fitness_best['Best fitness value'] = BestFitness

    fitness_mean = pd.DataFrame()
    fitness_mean['Mean fitness value'] = MeanFitness
    plot_fitness(BestFitness, MeanFitness)

    # Generate ranking of alternatives from test dataset using weights generated by DE algorithm `BestSolution`
    spotis = SPOTIS()
    pref = spotis(X_test, BestSolution, types, bounds)
    # Rank test alternatives according to preferences calculated by SPOTIS
    y_pred = rank_preferences(pref, reverse = False)
    # Calculate correlation between real ranking and predicted ranking of test alternatives using Spearman rank correlation coefficient
    print('\nRankings consistency: ', corrs.spearman(y_test, y_pred))

    # Save result ranking in dataframe and plot chart
    results = pd.DataFrame(index = test_df.index[:-1])
    results['Real rank'] = y_test
    results['Predicted rank'] = y_pred
    results.to_csv('results/results_de.csv')
    
    plot_rankings(results)
    

    '''
    # ====================================================================================
    # Comparative analysis
    # Other approaches determining weights
    # train_weights are real weights
    # de_weights are weights determined by DE algorithm
    # X_test is matrix to assess (dataset of products to assess)
    # y_test is real ranking
    dw = pd.read_csv('weights_DE.csv')
    de_weights = dw['DE weights'].to_numpy()

    results_benchmark = pd.DataFrame(index = test_df.index[:-1])
    results_benchmark['Real'] = y_test

    # DE approach
    spotis = SPOTIS()
    pref = spotis(X_test, de_weights, types, bounds)
    # Rank test alternatives according to preferences calculated by SPOTIS
    rank = rank_preferences(pref, reverse = False)
    results_benchmark['DE'] = rank

    # Entropy approach
    weights = pyweights.entropy_weighting(X_test)

    pref = spotis(X_test, weights, types, bounds)
    # Rank test alternatives according to preferences calculated by SPOTIS
    rank = rank_preferences(pref, reverse = False)
    results_benchmark['Entropy'] = rank

    # CRITIC approach
    weights = pyweights.critic_weighting(X_test)

    pref = spotis(X_test, weights, types, bounds)
    # Rank test alternatives according to preferences calculated by SPOTIS
    rank = rank_preferences(pref, reverse = False)
    results_benchmark['CRITIC'] = rank

    # Gini approach
    weights = pyweights.gini_weighting(X_test)

    pref = spotis(X_test, weights, types, bounds)
    # Rank test alternatives according to preferences calculated by SPOTIS
    rank = rank_preferences(pref, reverse = False)
    results_benchmark['Gini'] = rank
    results_benchmark.to_csv('results/comparative_analysis.csv')
    
    # Calculating correlations between rankings obtained with different weighting approaches
    method_types = list(results_benchmark.columns)

    dict_new_heatmap_rs = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rs.add(el, [])

    # heatmaps for calculated correlations coefficient
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rs[j].append(corrs.spearman(results_benchmark[i], results_benchmark[j]))
        
    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    df_new_heatmap_rs.to_csv('results/df_new_heatmap_rs.csv')
    # ===========================================================================================
    '''

    
if __name__ == '__main__':
    main()