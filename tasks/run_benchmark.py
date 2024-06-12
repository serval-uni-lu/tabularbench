from tabularbench.benchmark.benchmark import benchmark


def run():
    print("Welcome to TabularBench!")

    clean_acc, robust_acc = benchmark(
        dataset="URL",
        model="STG_Default",
        distance="L2",
        constraints=True,
    )
    print(f"Clean accuracy: {clean_acc}")
    print(f"Robust accuracy: {robust_acc}")


if __name__ == "__main__":
    run()
