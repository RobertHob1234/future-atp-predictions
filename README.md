# future-atp-predictions

Pure-NumPy feedforward neural network for ATP tennis match outcome prediction. Forward prop, backprop, and mini-batch SGD all implemented by hand (no PyTorch, no TensorFlow).

## Architecture

3-layer MLP, He-initialized:

```
9 inputs -> 128 ReLU -> 32 ReLU -> 1 sigmoid
```

Inputs: both players' ages, ATP ranks, rank points, plus categorical codes for surface (clay / grass / hard) and handedness (left / right). Output: probability that player 0 wins.

Trained with binary cross-entropy, batch size 256, 100 epochs. Standardization (`StandardScaler`) is fit on train and persisted to `model_checkpoints/scaler_params.npz` for reuse at predict time.

## Files

- `data_processing.py`: builds the feature matrix from raw ATP CSVs
- `train.py`: model, training loop, and checkpointing
- `Validation.py`: held-out evaluation
- `predict.py`: load weights and score new matches
- `dataset_encoded.csv`, `dataset_for_training.csv`: processed data

## Stack

Python, NumPy, pandas, scikit-learn (only for `StandardScaler`), matplotlib (for the loss curve).
