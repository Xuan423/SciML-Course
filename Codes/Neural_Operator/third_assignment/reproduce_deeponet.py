from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "DeepONet"
    output = root / "Output" / "ODE_Preds.mat"
    checkpoint = root / "checkpoint" / "model.index"
    print("DeepONet demo path:", root)
    print("Original entry:", root / "main.py")
    print("Framework: TensorFlow v1 compatibility APIs")
    print("phmbench note: TensorFlow is not installed in the current environment.")
    print("Existing output:", output, "exists=", output.exists())
    print("Existing checkpoint:", checkpoint, "exists=", checkpoint.exists())
    print("Use a TensorFlow-compatible environment to rerun: cd Codes/Neural_Operator/DeepONet && python main.py")


if __name__ == "__main__":
    main()
