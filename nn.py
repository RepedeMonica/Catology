import numpy as np
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from modeling import read_data_set

doc_excel = "input_extended.xlsx"
df = read_data_set(doc_excel)

features = df.drop(columns=['Race','Row.names','sex','Age','cats in the household','stays in','lives in'])
target = df['Race']

train_X, test_X, train_Y , test_Y = train_test_split(features, target, test_size=0.25, random_state=42)

#target = target.to_numpy()
def normalize_data(train_X, test_X):
    x_min = train_X.min(axis=0)
    x_max = train_X.max(axis=0)

    #print(f"x_min: {x_min}")
    #print(x_max)

    train_X = (train_X - x_min) / (x_max - x_min)
    test_X = (test_X - x_min) / (x_max - x_min)
    return train_X, test_X
#train_X, test_X = normalize_data(train_X.to_numpy(), test_X.to_numpy())

def encode(train_Y, test_Y):
    train_Y = train_Y - 1
    test_Y = test_Y - 1

    num_classes = len(np.unique(train_Y))

    train_Y_encoding = np.zeros((train_Y.size, num_classes))
    train_Y_encoding[np.arange(train_Y.size), train_Y] = 1

    test_Y_encoding = np.zeros((test_Y.size, num_classes))
    test_Y_encoding[np.arange(test_Y.size), test_Y] = 1
    return train_Y_encoding, test_Y_encoding
train_Y_encoding, test_Y_encoding = encode(train_Y.to_numpy(), test_Y.to_numpy())

np.random.seed(42)
def initialization(input_size, hidden_size1, hidden_size2, output_size):
    #weight intialization cu xavier uniform , slide 35 curs 6
    limit_W1 = np.sqrt(6/ (input_size + hidden_size1))
    W1 = np.random.uniform(-limit_W1, limit_W1, (input_size, hidden_size1))
    #W1 = np.random.randn(input_size, hidden_size1) * 0.01
    b1 = np.zeros((1, hidden_size1))

    limit_W2 = np.sqrt(6 / (hidden_size1 + hidden_size2))
    W2 = np.random.uniform(-limit_W2, limit_W2, (hidden_size1, hidden_size2))
    #W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
    b2 = np.zeros((1, hidden_size2))

    limit_W3 = np.sqrt(6 / (hidden_size2 + output_size))
    W3 = np.random.uniform(-limit_W3, limit_W3, (hidden_size2, output_size))
    #W3 = np.random.randn(hidden_size2, output_size) * 0.01
    b3 = np.zeros((1, output_size))

    return W1, b1, W2, b2, W3, b3

input_size = train_X.shape[1]
#output_size = train_Y_encoding.shape[1]
#print(f"Inputsize: {input_size}")
#print(f"Outputsize: {output_size}")
hidden_size1 = 128
hidden_size2 = 64
learning_rate = 0.1
epochs = 1000
#W1, b1, W2, b2, W3, b3 = initialization(input_size, hidden_size1, hidden_size2, output_size)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return z * (1 - z)

def relu(z):
    return np.maximum(0, z)
def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) #slide 30 curs 5
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(t_true, y_pred):
    return -np.mean(np.sum(t_true * np.log(y_pred), axis=1)) #slide 21 curs 5

def forward(X, W1, b1, W2, b2, W3, b3, dropout_rate = 0.1, training=True): #90% raman activi
    #slide 35 curs 4 sau pseudocodul generalizat-48 curs 5
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    if training:
        dropout_mask = (np.random.rand(*A1.shape) > dropout_rate).astype(float)
        A1 *= dropout_mask
        A1 /= (1 - dropout_rate)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backward(X, Y, A1, A2, A3, W1, b1, W2, b2, W3, b3, learning_rate):
    m = X.shape[0]

    #slide 36,curs5 -dem pt formula
    dZ3 = A3 - Y  #eroarea pt final layer, formula slide 44, curs5 #derivata softmax
    dW3 = np.dot(A2.T, dZ3)  / m   #formula slide 46 - formula basic de calculare a w
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m


    dA2 = np.dot(dZ3, W3.T)  #eroarea propagata, slide 45
    dZ2 = dA2 * relu_derivative(A2)  #eroarea finala= eroarea propagata*deriv relu
    # slide 6, curs 5
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    #slide 83, curs4
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    return W1, b1, W2, b2, W3, b3

def accuracy(t_true, y_pred):
    pred_label = np.argmax(y_pred, axis=1)
    true_label = np.argmax(t_true, axis=1)
    return np.mean(pred_label == true_label)

def train(X, Y, W1, b1, W2, b2, W3, b3, epochs, learning_rate, batch_size):
    patience = 35
    decay_rate = 0.2
    best_accuracy = 0
    epochs_without_improvement = 0

    m = X.shape[0]
    epoch = 0
    start_time = time.time()
    max_time = 120

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    while (time.time() - start_time) <= max_time:
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]

            Z1, A1, Z2, A2, Z3, A3 = forward(X_batch, W1, b1, W2, b2, W3, b3)
            W1, b1, W2, b2, W3, b3 = backward(X_batch, y_batch, A1, A2, A3, W1, b1, W2, b2, W3, b3, learning_rate)

        _, _, _, _, _, A3_train = forward(X, W1, b1, W2, b2, W3, b3)
        _, _, _, _, _, A3_test = forward(test_X, W1, b1, W2, b2, W3, b3 , training=False)

        train_loss = cross_entropy(Y, A3_train)
        test_loss = cross_entropy(test_Y_encoding, A3_test)
        train_acc = accuracy(Y, A3_train)
        test_acc = accuracy(test_Y_encoding, A3_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1} - Train Accuracy: {train_acc*100:.2f}% - Test Accuracy: {test_acc*100:.2f}%  - Test Loss: {test_loss:.4f} - Elapsed Time: {elapsed_time:.2f}")

        if elapsed_time >= max_time:
            print("Stop due to time limit")
            break

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            learning_rate = learning_rate * decay_rate
            epochs_without_improvement = 0
            print(f"Reducing learning rate to {learning_rate}")
        epoch +=1

    return W1, b1, W2, b2, W3, b3 , train_losses, test_losses, train_accuracies, test_accuracies
#W1_trained, b1_trained, W2_trained, b2_trained, W3_trained, b3_trained, train_losses, test_losses, train_accuracies, test_accuracies = train(train_X, train_Y_encoding, W1, b1, W2, b2, W3, b3, epochs, learning_rate, batch_size=64)

def save_weights_to_text(weights, biases, file_name):
    with open(file_name, "w") as file:
        for i, (W, b) in enumerate(zip(weights, biases), start=1):
            file.write(f"Weights Layer {i}:\n")
            np.savetxt(file, W, fmt="%.6f")
            file.write(f"Biases Layer {i}:\n")
            np.savetxt(file, b, fmt="%.6f")
            file.write("\n")

#weights = [W1_trained, W2_trained, W3_trained]
#biases = [b1_trained, b2_trained, b3_trained]
#save_weights_to_text(weights, biases, "neural_network_weights.txt")

def check_accuracy_nn(W1_trained, b1_trained, W2_trained, b2_trained, W3_trained, b3_trained):
    _,_,_,_,_, test_pred_final = forward(test_X, W1_trained, b1_trained, W2_trained, b2_trained, W3_trained, b3_trained, training =False)
    final_test_accuracy = accuracy(test_Y_encoding, test_pred_final)
    print(f'Final test Accuracy: {final_test_accuracy*100:.2f}%')

    final_test_predictions = np.argmax(test_pred_final, axis=1)
    true_labels = np.argmax(test_Y_encoding, axis=1)

    print("\nClassification Report:")
    print(classification_report(true_labels, final_test_predictions))


