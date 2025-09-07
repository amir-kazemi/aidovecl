import argparse
from src import aidovecl, torch_models, yolo, prompt_semantics

def main(args):
    try:
        if args.option == "aidovecl":
            aidovecl.run()
        elif args.option == "finetune_torch":
            torch_models.finetune_all()
        elif args.option == "test_torch":
            torch_models.test_all()
        elif args.option == "finetune_yolo":
            yolo.finetune_all()
        elif args.option == "test_yolo":
            yolo.test_all()
        elif args.option == "prompt_semantics":
            prompt_semantics.save_results()
    except Exception as e:
        print(f"[ERROR] Failed to execute {args.option}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dataset operations, training, or testing routines"
    )
    parser.add_argument(
        "option",
        choices=[
            "aidovecl",
            "finetune_torch", "test_torch",
            "finetune_yolo", "test_yolo",
            "prompt_semantics"
        ],
        help="Operation or routine to execute"
    )
    args = parser.parse_args()
    main(args)