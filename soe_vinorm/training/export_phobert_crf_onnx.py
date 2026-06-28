import argparse
import json

from soe_vinorm.phobert_crf import export_phobert_crf_to_onnx


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export-phobert-crf-onnx",
        description="Export a trained PhoBERT+CRF NSW detector to ONNX.",
    )
    parser.add_argument("--model-dir", required=True, help="Trained model directory.")
    parser.add_argument(
        "--output",
        help="Output ONNX file path. Defaults to <model-dir>/model.onnx.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    output_path = export_phobert_crf_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        opset=args.opset,
        device=args.device,
    )
    print(json.dumps({"onnx_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
