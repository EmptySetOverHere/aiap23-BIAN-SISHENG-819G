from src.preprocess import load_config
from src.train import train_and_evaluate


def main():
    config = load_config("config.yml")
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
