import csv
import numpy as np

#Softmax function used to quantize the values between 0 and 1
def Softmax(layer):
    e_x = np.exp(layer - np.max(layer))
    return e_x/e_x.sum()

class PolicyNeuralNetTest:
    def __init__(self, csv_path=None):
        INPUT_SIZE = 8
        HIDDEN_LAYER_1_SIZE = 50
        HIDDEN_LAYER_2_SIZE = 70
        OUTPUT_SIZE = 11

        self.LR = 0.01
        self.w1 = np.random.randn(HIDDEN_LAYER_1_SIZE, INPUT_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_LAYER_1_SIZE)
        self.w2 = np.random.randn(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_1_SIZE)
        self.b2 = np.zeros(HIDDEN_LAYER_2_SIZE)
        self.w3 = np.random.randn(OUTPUT_SIZE, HIDDEN_LAYER_2_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_2_SIZE)
        self.b3 = np.zeros(OUTPUT_SIZE)

        if csv_path is not None:
            self.load_from_csv(csv_path)

    def forward(self, input_layer):
        z1 = self.w1 @ input_layer + self.b1
        a1 = np.tanh(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = np.tanh(z2)
        z3 = self.w3 @ a2 + self.b3
        a3 = Softmax(z3)
        return a3

    def load_from_csv(self, csv_path: str):
        buffers = {
            "w1": np.zeros_like(self.w1, dtype=float),
            "b1": np.zeros_like(self.b1, dtype=float),
            "w2": np.zeros_like(self.w2, dtype=float),
            "b2": np.zeros_like(self.b2, dtype=float),
            "w3": np.zeros_like(self.w3, dtype=float),
            "b3": np.zeros_like(self.b3, dtype=float),
        }
        seen = {k: 0 for k in buffers.keys()}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                p = r["param"].strip()
                if p not in buffers:
                    continue
                row = int(r["row"])
                val = float(r["value"])
                col_raw = (r.get("col") or "").strip()

                if p.startswith("b"):
                    if row < 0 or row >= buffers[p].shape[0]:
                        raise IndexError(f"{p} row {row} out of range")
                    buffers[p][row] = val
                else:
                    if col_raw == "":
                        raise ValueError(f"Missing 'col' for matrix param {p} at row {row}")
                    col = int(col_raw)
                    if row < 0 or row >= buffers[p].shape[0] or col < 0 or col >= buffers[p].shape[1]:
                        raise IndexError(f"{p} index {(row, col)} out of range")
                    buffers[p][row, col] = val

                seen[p] += 1

        expected_counts = {
            "w1": self.w1.size, "b1": self.b1.size,
            "w2": self.w2.size, "b2": self.b2.size,
            "w3": self.w3.size, "b3": self.b3.size,
        }
        for k in buffers:
            if seen[k] == 0:
                raise ValueError(f"No entries for {k} found in CSV.")
            if seen[k] != expected_counts[k]:
                print(f"Warning: {k} got {seen[k]} entries; expected {expected_counts[k]}.")
                pass

        # Assign into the model
        self.w1, self.b1 = buffers["w1"], buffers["b1"]
        self.w2, self.b2 = buffers["w2"], buffers["b2"]
        self.w3, self.b3 = buffers["w3"], buffers["b3"]

