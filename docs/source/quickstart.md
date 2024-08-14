# Quickstart

The easiest way to run the benchmark is to to use a pre-trained model.
The models files can be [downloaded here](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/thibault_simonetto_uni_lu/EvkG4BI0EqJFu436biA2C_sBpkEKTTjA5PgZU_Z9jwNNSA?e=62a4Dm)
and should be placed *data/models* folder

```python
    from tabularbench.benchmark.benchmark import benchmark
    clean_acc, robust_acc = benchmark(
        dataset="URL",
        model="STG_Default",
        distance="L2",
        constraints=True,
    )
    print(f"Clean accuracy: {clean_acc}")
    print(f"Robust accuracy: {robust_acc}")
```