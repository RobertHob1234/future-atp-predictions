from train import predict_from_file
import numpy as np

if __name__ == "__main__":
    feature_cols = [
        'player_0_age', 'player_1_age',
        'player_0_rank', 'player_0_rank_points',
        'player_1_rank', 'player_1_rank_points'
    ]

    data = np.load("model_checkpoints/scaler_params.npz")
    mean, scale = data["mean"], data["scale"]  # each shape (6,)

    raw = {
        "player_0_age": 69.1,
        "player_1_age": 60.9,
        "player_0_rank": 38.0,
        "player_0_rank_points": 1130.0,
        "player_1_rank": 42.0,
        "player_1_rank_points": 1040.0,
        "surface_code": 0,
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

    A2, preds = predict_from_file(r"D:\Weights_Bias\final_lr0.5000.npz",x) # (9,)

    if preds == 1:
        print("Based on your input the model predicted that player 0 would win")
    else:
        print("Based on your input the model predicted that player 1 would win")