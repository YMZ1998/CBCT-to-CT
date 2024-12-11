import os

import numpy as np
import onnx
import onnxruntime
import torch


class Model(torch.nn.Module):
    def __init__(self, stage1, stage2, resbranch):
        super(Model, self).__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.resbranch = resbranch

    def load_model_weights(self, weight_path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(weight_path, weights_only=False, map_location='cpu')
        self.stage1.load_state_dict(checkpoint['model_stage1'])
        self.stage2.load_state_dict(checkpoint['model_stage2'])
        self.resbranch.load_state_dict(checkpoint['model_resbranch'])

    def forward(self, x, mask):
        x1 = self.stage1(x * mask)
        x1_detached = x1.clone().detach()
        x2 = self.stage2(x1_detached * mask)
        x3 = self.resbranch(x * mask)
        return torch.tanh(x2 + x3)


def to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array."""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_onnx(args):
    """Convert PyTorch model to ONNX format."""
    from parse_args import get_model, get_best_weight_path

    # Prepare device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Initialize model
    stage1, stage2, resbranch = get_model(args)
    model = Model(stage1, stage2, resbranch).to(device)

    # Load weights
    weights_path = get_best_weight_path(args)
    model.load_model_weights(weights_path)

    # Set model to evaluation mode
    model.eval()

    # Create dummy input and mask
    batch_size = 1
    x = torch.rand(batch_size, 5, args.image_size, args.image_size, requires_grad=True).to(device)
    mask = torch.rand(batch_size, 1, args.image_size, args.image_size, requires_grad=True).to(device)

    # Perform forward pass with PyTorch
    torch_out = model(x, mask)

    # Define the output ONNX file name
    onnx_file_name = os.path.join("checkpoint", f"{args.arch}_best_model.onnx")

    # Export model to ONNX
    torch.onnx.export(model, (x, mask), onnx_file_name,
                      input_names=["input", "mask"], output_names=["output"],
                      verbose=True)

    # Validate the exported ONNX model
    try:
        onnx_model = onnx.load(onnx_file_name)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model is valid!")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        return

    # Run ONNX model with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_file_name)
    for input in ort_session.get_inputs():
        print(f"Input name: {input.name}, shape: {input.shape}")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x), ort_session.get_inputs()[1].name: to_numpy(mask)}

    # Compare PyTorch and ONNX outputs
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # Save the ONNX model
    print(f"Model saved as ONNX to {onnx_file_name}.")


if __name__ == '__main__':
    # Assume parse_args() is already defined elsewhere, so we pass it as an argument
    from parse_args import parse_args

    args = parse_args()
    convert_onnx(args)
