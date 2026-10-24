import argparse
import uvicorn
from server import app


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--batch-wait-s", type=float, default=0.05)

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--batch", action="store_true")
    modes.add_argument("--spec-decode", action="store_true")

    args = parser.parse_args()

    if args.max_batch_size < 1 or args.batch_wait_s < 0:
        parser.error("Batch size must be positive and wait must be nonnegative.")

    app.state.config = args
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()