from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "FNO_earthquake1D"
    print("FNO earthquake 1D demo path:", root)
    print("Original entry:", root / "fourier_1d.py")
    print("Framework: PyTorch")
    print("Reference data:", root / "Data" / "eq_data_N100_r2500.mat")
    print("The main homework implementation extends this demo's spectral layer idea to 2D Cavity Flow.")
    print("To rerun the original demo: cd Codes/Neural_Operator/FNO_earthquake1D && python fourier_1d.py")


if __name__ == "__main__":
    main()
