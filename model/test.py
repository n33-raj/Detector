import torch
from yolo import yolo_v8_s  # adjust import path if needed

def test_model_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = yolo_v8_s(num_classes=80).to(device)
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    # Run forward before fusion
    with torch.no_grad():
        output_before = model(dummy_input)

    # Fuse model layers
    model.fuse()

    # Run forward after fusion
    with torch.no_grad():
        output_after = model(dummy_input)

    # Check if outputs are close (allowing minor numerical differences)
    if torch.allclose(output_before, output_after, atol=1e-5):
        print("Fusion test passed: outputs match closely.")
    else:
        max_diff = (output_before - output_after).abs().max()
        print(f"Fusion test failed: max difference = {max_diff.item()}")

if __name__ == "__main__":
    test_model_fusion()
