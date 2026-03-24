from pathlib import Path


def main() -> None:
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    print("Prepare your structured panel and transcript-aligned features here.")


if __name__ == "__main__":
    main()
