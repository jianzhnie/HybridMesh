"""Allow `python -m hpmesh` to run the training entry point."""

from .trainer.train import main

if __name__ == "__main__":
    main()
