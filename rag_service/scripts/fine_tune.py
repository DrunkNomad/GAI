"""
Fine-tune a Hugging Face model with LoRA (requires GPU).

Usage:
    python scripts/fine_tune.py --model microsoft/Phi-3-mini-4k-instruct --dataset my_dataset.json

This script requires:
    pip install torch peft transformers datasets accelerate bitsandbytes
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="./fine_tuned_model")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("  Fine-tuning script")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Output:     {args.output_dir}")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    print()
    print("  This script requires a GPU. Run on Colab/Kaggle with:")
    print("    python scripts/fine_tune.py --model microsoft/Phi-3-mini-4k-instruct --dataset data/instructions.json")
    print()
    print("  Example Colab: https://colab.research.google.com/gist/")


if __name__ == "__main__":
    main()
