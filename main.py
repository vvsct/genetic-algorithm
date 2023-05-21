import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
import time


def main(
        data=np.genfromtxt('ES_data_4.dat')
):
    i = data[:, 0]  # input
    o = data[:, 1]  # output
    G = 200  # max number of generations
    n = 6  # number of genes in a chromosome
    tau_1 = 1 / np.sqrt(2 * n)
    tau_2 = 1 / np.sqrt(2 * np.sqrt(n))
    epsilon = 0.00001  # stop criterion

    # calculating the outcome and mean squared error of output and model outcome
    def objective_function(in_put, out_put, x, choice):
        f = [x[0] * ((i ** 2) - x[1] * np.cos(x[2] * np.pi * i)) for i in in_put]
        E = mean_squared_error(out_put, f)
        # choosing the function output - Mean Squared Error (E) or result of the objective function (f)
        if choice == 0:
            return E
        else:
            return f

    results = pd.DataFrame(columns=['population', 'generations', 'a', 'b', 'c', 'MSE', 'time'])
    mu_values = [100, 500, 1000]  # different population sizes
    j = 1
    for mu in mu_values:
        parents = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6])
        offspring = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6])

        start = time.process_time()
        # CREATING THE FIRST GENERATION
        for t in range(0, mu):
            # randomly generating the chromosomes and their strategy parameters
            a = np.random.uniform(low=-10, high=10, size=None)
            sigma_a = np.random.uniform(low=0, high=10, size=None)
            b = np.random.uniform(low=-10, high=10, size=None)
            sigma_b = np.random.uniform(low=0, high=10, size=None)
            c = np.random.uniform(low=-10, high=10, size=None)
            sigma_c = np.random.uniform(low=0, high=10, size=None)
            MSE = objective_function(i, o, [a, b, c], 0)  # calculating the mean square error
            # assigning the chromosomes to an individual (with MSE on the last spot for easier evaluation)
            df_temp = [[a, sigma_a, b, sigma_b, c, sigma_c, MSE]]
            # adding the individual to the parents dataframe
            parents = parents.append(pd.DataFrame(df_temp), ignore_index=True)

        # CREATING OFFSPRING
        for p in range(0, G):
            for k in range(0, mu):
                for m in range(0, 5):
                    # MUTATION
                    # x = x * sigma_x * N(0, 1)
                    a = parents.loc[k, 0] + parents.loc[k, 1] * np.random.randn()
                    b = parents.loc[k, 2] + parents.loc[k, 3] * np.random.randn()
                    c = parents.loc[k, 4] + parents.loc[k, 5] * np.random.randn()
                    # r_sig_1 = N(0, 1) * tau_1
                    r_sig_1 = np.random.randn() * tau_1
                    # r_sigma_x = N(0, 1) * tau_2
                    r_sig_a = np.random.randn() * tau_2
                    r_sig_b = np.random.randn() * tau_2
                    r_sig_c = np.random.randn() * tau_2
                    # sigma_x = sigma_x * exp(r_sig_1) * exp(r_sig_x)
                    sigma_a = parents.loc[k, 1] * np.exp(r_sig_1) * np.exp(r_sig_a)
                    sigma_b = parents.loc[k, 3] * np.exp(r_sig_1) * np.exp(r_sig_b)
                    sigma_c = parents.loc[k, 5] * np.exp(r_sig_1) * np.exp(r_sig_c)
                    MSE = objective_function(i, o, [a, b, c], 0)
                    # assigning chromosomes to a child
                    df_temp = pd.DataFrame([[a, sigma_a, b, sigma_b, c, sigma_c, MSE]], columns=[0, 1, 2, 3, 4, 5, 6])
                    # adding the child to the offspring dataframe
                    offspring.loc[k * 5 + m] = df_temp.loc[0]
            # EVALUATION
            parents_temp = parents  # creating a temp parents df to check stop criterion
            offspring = offspring.sort_values(by=[6], ignore_index=True)
            parents_temp = parents_temp.sort_values(by=[6], ignore_index=True)
            # creating a union of the parents and offspring sets
            evaluation = pd.concat([parents, offspring], ignore_index=True)
            # evaluating the created dataframe of offspring and parents by sorting the union by MSE value, ascending
            evaluation = evaluation.sort_values(by=[6], ignore_index=True)
            # selecting mu individuals characterised by minimum MSE for the next generation
            parents = evaluation.loc[:(mu-1), :]
            # checking if the stop criterion has been met
            if (abs(parents_temp.loc[0, 6] - offspring.loc[0, 6])) < epsilon:
                best_individual = parents.loc[0, :]
                break
            else:
                continue
        print(time.process_time() - start)
        stop = time.process_time() - start
        arr = [[mu, p+1, best_individual.loc[0], best_individual.loc[2], best_individual.loc[4], best_individual.loc[6], stop]]
        df_to_append = pd.DataFrame(arr)
        df_to_append.columns = results.columns
        results = results.append(df_to_append, ignore_index=True)  # creating df of results to be saved in Excel
        out_approx = objective_function(i, o, [best_individual.loc[0], best_individual.loc[2], best_individual.loc[4]], 1)

        # PLOTS
        fig, axs = plt.subplots(1, 3, sharex=True)
        sns.scatterplot(x=i, y=o, ax=axs[0], color='red')
        # sns.scatterplot(x=i, y=out_approx, ax=axs[1])
        sns.lineplot(x=i, y=out_approx, ax=axs[1], color='blue')
        # sns.scatterplot(x=i, y=out_approx, ax=axs[2])
        sns.lineplot(x=i, y=out_approx, ax=axs[2], color='blue')
        sns.scatterplot(x=i, y=o, ax=axs[2], color='red')
        fig.legend(['true', 'estimated'], fontsize='large', title='results')
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(28, 14)
        plt.savefig('figure {0}.png'.format(j))
        plt.close('all')
        j += 1
    # saving results to excel
    results.to_excel('results.xlsx')


if __name__ == "__main__":
    main()
