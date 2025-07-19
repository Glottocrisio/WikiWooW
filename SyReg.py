#Script for performing symbolic regression using PySR and gplearn

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pysr import PySRRegressor

def symbolicregressor(dataset):
    df = pd.read_csv(dataset, sep=';')

    X = df.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 13]].values 
    y = df.iloc[:, 21].values

    #X = df.iloc[:, [2, 5, 6, 7, 8, 9]].values 
    #y = df.iloc[:, 13].values

    # Instantiate and configure the PySRRegressor
    model = PySRRegressor(
        niterations=5,       # Number of iterations of the search
        populations=5,       # Number of populations to maintain
        ncyclesperiteration=100,
        binary_operators=["+", "*", "/", "-"],  # Allowed operators
        unary_operators=[
            "sin", "cos", "exp", "log"  # Mathematical functions to consider
        ],
        verbosity=1          # Show progress of the algorithm
    )

    # Fit the model
    model.fit(X, y)

    # Print the best equation found
    #print("Best equation:", model.equations_[0])
    # Plot the original data and the fitted model
    # plt.scatter(X[:, 5], y, color='blue', label='Data')
    # x_range = np.linspace(0, 6, 100)
    # predictions = model.predict(x_range[:, np.newaxis])
    # plt.plot(x_range, predictions, color='red', label='Model fit')
    # plt.legend()
    # plt.show()
    print(model.equations_)

#symbolicregressor("updated_data.csv")

def symbolicregressor2(dataset):
    df = pd.read_csv(dataset, sep=';')
    df = df.dropna()
    X = df.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 13]].values 
    y = df.iloc[:, 21].values
    #add(inv(mul(div(neg(inv(0.893)), mul(sin(X1), neg(X0))), neg(div(abs(0.715), mul(-0.795, X0)))))
    #add(abs(abs(sub(neg(sqrt(mul(sin(sin(abs(0.396))), neg(X0)))), neg(mul(X1, X0))))), -0.268))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit a symbolic regressor
    model = SymbolicRegressor(population_size=5000,
                              generations=20, 
                              stopping_criteria=0.01,
                              p_crossover=0.7, 
                              p_subtree_mutation=0.1,
                              p_hoist_mutation=0.05,
                              p_point_mutation=0.1,
                              max_samples=0.9,
                              verbose=1,
                              function_set=('add', 'sub', 'mul', 'div', 'sin', 'log', 'sqrt', 'abs', 'neg', 'inv'),
                              metric='mse',
                              parsimony_coefficient=0.01,  # Controls bloat
                              random_state=0)

    model.fit(X_train, y_train)

    # Print the discovered expression
    print("Discovered expression:")
    print(model._program)

    # Evaluate model performance
    print("\nModel performance:")
    print("Training score:", model.score(X_train, y_train))
    print("Testing score:", model.score(X_test, y_test))

    # Plotting actual vs predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, model.predict(X_test), edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


from gplearn.genetic import SymbolicRegressor


#symbolic regressor
def symbreg(dataset):
    # Example data
    df = pd.read_csv(dataset, sep=';')
    print(df.columns)
    df = df.dropna()
    X = df.iloc[:, [19, 20]].values 
    y = df.iloc[:, 21].values

    #X = df.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 13]].values 
    #y = df.iloc[:, 21].values

    # Initialize and fit the model
    model = SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1)
    model.fit(X, y)

    # Display model
    print("Expression:", model._program)
    return model._program

#symbreg("enhanced_entity_pairs.csv")
symbolicregressor2("enhanced_entity_pairs_2.csv")
symbolicregressor2("enhanced_entity_pairs_3.csv")
#ymbolicregressor2("updated_data.csv")
