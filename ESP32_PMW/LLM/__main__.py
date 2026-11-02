"""`python -m LLM self-check`: every offline test. No camera, serial or coil drive."""
import sys

if __name__ == "__main__":
    if sys.argv[1:] != ["self-check"]:
        sys.exit("usage: python -m LLM self-check   (the workflow itself is LLM.bench.Bench)")
    from LLM.tests.run import run
    run()
    print("All offline checks passed")
