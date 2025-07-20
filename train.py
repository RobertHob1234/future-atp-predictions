"""As you'd probably guess, this file trains the model
    Robert Hoang
    2025-07-18"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt # lol  I plotted to find the learning rate but didn't really use otherwise

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def d_tanh(z):
    # derivative of tanh is 1 – tanh(Z)^2 i think
    A = np.tanh(z)
    return 1 - np.power(A, 2)

def relu(z):
    return np.maximum(0, z)

def d_relu(z):
    return (z > 0).astype(float)

def vectorize_data():
    # load CSV into a DataFrame
    df = pd.read_csv(r"D:\tennis_processed\dataset_encoded.csv")

    # define which columns are inputs
    feature_cols = [
        'player_0_age', 'player_1_age',
        'player_0_rank', 'player_0_rank_points',
        'player_1_rank', 'player_1_rank_points'
    ]
    # output
    label_col = 'Output'

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    os.makedirs("model_checkpoints", exist_ok=True)
    np.savez(r"model_checkpoints/scaler_params.npz",
             mean=scaler.mean_,
             scale=scaler.scale_)
    print("Saved scaler parameters to model_checkpoints/scaler_params.npz")

    # extract into NumPy arrays
    X = df[feature_cols + ['surface_code','p0_hand_code','p1_hand_code']].to_numpy().T   # shape: (9,m)
    Y = df[label_col].to_numpy()  # shape: (m,)
    Y = Y.reshape(1, -1)

    # check
    """
    print("X.shape:", X.shape)  #  (215000, 9)
    print("y.shape:", Y.shape)  #  (1, 215000)
    print(df[feature_cols + ['surface_code','p0_hand_code','p1_hand_code']].head())
    print(df[label_col].head())
    """
    return X, Y

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h1 = 128
    n_h2 = 32
    n_y = Y.shape[0]

    return n_x, n_h1, n_h2, n_y


def init_weight_and_biases(n_x, n_h1, n_h2, n_y):

    # use He-initialization for relu.
    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h1, 1)) #col
    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2.0 / n_h1)
    b2 = np.zeros((n_h2, 1)) #col
    W3 = np.random.randn(n_y, n_h2) * np.sqrt(2.0 / n_h2)
    b3 = np.zeros((n_y, 1))  # col

    weights_biases = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                    "b3": b3
                    }

    return weights_biases


def make_batches(X, Y, bs):
    m = X.shape[1]
    perm = np.random.permutation(m)
    batches = []
    # full batches
    for start in range(0, m, bs):
        idx = perm[start:start+bs]
        Xb = X[:, idx]   # input feature size * batch example size
        Yb = Y[:, idx]   # output size (1) * batch example size
        batches.append((Xb, Yb))
    return batches

def for_prop(X, weights_and_biases):

    W1 = weights_and_biases["W1"]
    b1 = weights_and_biases["b1"]
    W2 = weights_and_biases["W2"]
    b2 = weights_and_biases["b2"]
    W3 = weights_and_biases["W3"] #(32, 128)
    b3 = weights_and_biases["b3"]

    Z1 = np.dot(W1, X) + b1  # (7,m)
    A1 = relu(Z1)  # (7,m)
    Z2 = np.dot(W2, A1) + b2  # (1,m)
    A2 = relu(Z2)  # (128,m)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)


    for_prop_values = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }

    return for_prop_values


def compute_cost(A3, Y):

    m = Y.shape[1]

    logprobs = Y * np.log(A3) + (1 - Y) * np.log(1 - A3)
    cost = -1 / m * np.sum(logprobs, axis=1, keepdims=True)

    cost = float(np.squeeze(cost))

    return cost

def back_prop(weights_and_biases, for_prop_values, X, Y):

    m = X.shape[1]

    Z1 = for_prop_values["Z1"]
    Z2 = for_prop_values["Z2"]

    W1 = weights_and_biases["W1"]
    W2 = weights_and_biases["W2"]
    W3 = weights_and_biases["W3"]
    A1 = for_prop_values["A1"]
    A2 = for_prop_values["A2"]
    A3 = for_prop_values["A3"]

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3,A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = d_relu(Z2) * np.dot(W3.T,dZ3) # (n_h2, m)
    dW2 = (1 / m) * np.dot(dZ2,A1.T)  # (n_h2, n_h1)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = d_relu(Z1) * np.dot(W2.T, dZ2)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3
             }

    """"print(f"dZ2 values:\n{dZ2}")
    print(f"dW2 values:\n{dW2}")
    print(f"db2 values:\n{db2}")
    print(f"dZ1 values:\n{dZ1}")
    print(f"dW1 values:\n{dW1}")
    print(f"db1 values:\n{db1}"""

    return grads

def train_model(learning_rate):
    #learning_rate = 0.0015

    X, Y = vectorize_data()

    # ________________Trying with whole set as one batch for personal interest_______________

    """n_x, n_h, n_y = layer_sizes(X, Y)
    # random initialization for weights and zeros for Other shit
    weights_biases = init_weight_and_biases(n_x, n_h, n_y)

    W1 = weights_biases["W1"]
    b1 = weights_biases["b1"]
    W2 = weights_biases["W2"]
    b2 = weights_biases["b2"]

    for_prop_values = for_prop(X, weights_biases)

    A1 = for_prop_values["A1"]
    Z1 = for_prop_values["Z1"]
    A2 = for_prop_values["A2"]
    Z2 = for_prop_values["Z2"]

    compute_cost(A2, Y)"""

    # _______________________________________________________________________________________


    n_x, n_h1, n_h2, n_y = layer_sizes(X, Y)
    # random initialization for weights and zeros for Other shit
    weights_biases = init_weight_and_biases(n_x, n_h1, n_h2, n_y)

    W1 = weights_biases["W1"]
    b1 = weights_biases["b1"]
    W2 = weights_biases["W2"]
    b2 = weights_biases["b2"]
    W3 = weights_biases["W3"]
    b3 = weights_biases["b3"]

    costs = []
    for it in range(100):
        epoch_cost = 0
        # takes x and y and randomizes order
        batches = make_batches(X, Y, 256)
        for i, (Xb, Yb) in enumerate(batches):
            m = len(batches)
            """print(f"Batch NO#: {i+1}")
            print(f"Xb.shape: {Xb.shape}")
            print(f"Yb.shape: {Yb.shape}")
            print(f"W1.shape: {W1.shape}")
            print(f"b1.shape: {b1.shape}")
            print(f"W2.shape: {W2.shape}")
            print(f"b2.shape: {b2.shape}")"""

            for_prop_values = for_prop(Xb, weights_biases)

            A1 = for_prop_values["A1"]
            Z1 = for_prop_values["Z1"]
            A2 = for_prop_values["A2"]
            Z2 = for_prop_values["Z2"]
            A3 = for_prop_values["A3"]
            Z3 = for_prop_values["Z3"]

            """print(f"Z1.shape: {Z1.shape}")
            print(f"A1.shape: {A1.shape}")
            print(f"Z2.shape: {Z2.shape}")
            print(f"A2.shape: {A2.shape}")"""

            grads = back_prop(weights_biases, for_prop_values, Xb, Yb)

            dW1 = grads["dW1"]
            db1 = grads["db1"]
            dW2 = grads["dW2"]
            db2 = grads["db2"]
            dW3 = grads["dW3"]
            db3 = grads["db3"]

            W1 -= dW1 * learning_rate
            b1 -= db1 * learning_rate
            W2 -= dW2 * learning_rate
            b2 -= db2 * learning_rate
            W3 -= dW3 * learning_rate
            b3 -= db3 * learning_rate

            """print(f"W1 values:\n{W1}")
            print(f"b1 values:\n{b1}")
            print(f"W2 values:\n{W2}")
            print(f"b2 values:\n{b2}")"""

            cost = compute_cost(A3, Yb)
            epoch_cost += cost

            if (i+1) % len(batches) == 0 and i != 0:
                epoch_cost /= m
            if (it == 0 and i ==0):
                print(epoch_cost)
                costs.append(epoch_cost)
        if (it) % 99 == 0:
            print(epoch_cost)

        costs.append(epoch_cost)


    os.makedirs("model_checkpoints", exist_ok=True)
    fname = rf"D:\Weights_Bias\final_lr{learning_rate:.4f}.npz"
    np.savez(fname,
             W1=W1, b1=b1,
             W2=W2, b2=b2,
             W3=W3, b3=b3)
    return costs

# stored cost around 0.61. Could be better but likely doesn't affect accuracy. Just based on random initialization
def load_weights(path):

    data = np.load(path)
    return {k: data[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}


def predict_from_file(weights_path, X, threshold=0.5):
    wb = load_weights(weights_path)
    W1, b1 = wb["W1"], wb["b1"]
    W2, b2 = wb["W2"], wb["b2"]
    W3, b3 = wb["W3"], wb["b3"]

    # First layer with relu. had tanh first but relu takes less time to compute
    Z1 = np.dot(W1,X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2,A1) + b2
    A2 = relu(Z2)
    # output
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    preds = (A3 > threshold).astype(int)
    return A3, preds

if __name__ == "__main__":
    costs = train_model(1.0)
    epochs = list(range(1, len(costs) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, costs, marker='o')
    plt.xlabel("epoch")
    plt.ylabel("Cost")
    plt.title("Training Cost Over Time")
    plt.grid(True)
    plt.show()


