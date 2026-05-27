from train import predict_from_file
import numpy as np

if __name__ == "__main__":
    feature_cols = [
        'player_0_age', 'player_1_age',
        'player_0_rank', 'player_0_rank_points',
        'player_1_rank', 'player_1_rank_points'
    ]

    data = np.load(r"C:\Users\rober\Documents\NeuralNetwork1\model_checkpoints\scaler_params.npz")
    mean, scale = data["mean"], data["scale"]  # each shape (6,)

    player_0 = "Fransisco Comesana"
    player_1 = "Alexander Zverev"

    raw = {
        "player_0_age": 25.11,
        "player_1_age": 28.25,
        "player_0_rank": 73.0,
        "player_0_rank_points": 861.0,
        "player_1_rank": 3.0,
        "player_1_rank_points": 6310.0,
        "surface_code": 1,
        "p0_hand_code": 0,
        "p1_hand_code": 0
    }

    # scaling
    cont_names = feature_cols
    raw_vals = np.array([raw[n] for n in cont_names], dtype=float)  # (6,)
    cont_scaled = (raw_vals - mean) / scale  # (6,)

    # full feature vector (9×1)
    x = np.vstack([
        cont_scaled.reshape(-1, 1),  # (6,1)
        np.array([[raw["surface_code"]],
                  [raw["p0_hand_code"]],
                  [raw["p1_hand_code"]]])  # (3,1)
    ])

    A3, preds = predict_from_file(r"D:\Weights_Bias\final_lr0.5000.npz",x) # (9,)

    if preds == 1:
        print(f"Based on your input the model predicted that {player_0} would win")
        print(f"Confidence = {A3 * 100}%")
    else:
        print(f"Based on your input the model predicted that {player_1} would win")
        print(f"Confidence = {(1-A3) * 100}")