"""testing/finding accuracy over examples of data
    Robert Hoang
    2025-07-19"""

from train import predict_from_file, vectorize_data

"""Ngl, this is evaluating on the training set rather than a held-out test set. With 215,000 unique 
matches however, I expect it to still capture most real world variation so I don't think it matters too much"""
def test_accuracy(nexamples):
    # vectorize data pulls it straight form encode_data. I'm doing this so I don't have to rescale myself
    X, Y = vectorize_data()  # X (9, m), Y (1, m)
    Xn = X[:, :nexamples]  # (9, nexamples)
    Yn = Y[0, :nexamples].astype(int)

    A3, predsn = predict_from_file(r"D:\Weights_Bias\final_lr0.1000.npz", Xn)  # shape (9,)

    correct = 0
    total = 0

   # computes accuracy
    for i in range(nexamples):
        print(f"Example {i+1:{len(str(nexamples))}d}: Predicted = {predsn[0, i]},  Actual = {Yn[i]}")

        if predsn[0, i] == Yn[i]:
            correct += 1
        total += 1

    print(f"{correct}/{total} ~ {round(correct / total * 100, 2)}% training accuracy")

if __name__ == "__main__":
    test_accuracy(213000)


