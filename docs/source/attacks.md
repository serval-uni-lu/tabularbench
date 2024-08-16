# Attacks

## Supported attacks

```{eval-rst}
.. include:: attacks_ref.rst
```

## Building a new attack

Attacks from TabularBench follow the same structre as attacks from [TorchAttacks.](https://github.com/Harry24k/adversarial-attacks-pytorch)
A new attack should then extend *torchattacks.attack.Attack*. 

To evaluate success rate with constraint satisfaction in the new attack, call *tabularbench.attacks.objective_calculator.ObjectiveCalculator*:

To evaluate individual constraint losses, call *tabularbench.constraints.constraints_backend_executor.ConstraintsExecutor*

For a complete example, refer to the implementation of [CAPGD](https://github.com/serval-uni-lu/tabularbench/blob/main/tabularbench/attacks/cpgd/cpgd.py)

## Submitting a new attack

We welcome attacks contributions that bring additional insights or challenges to the tabular adversarial robustness community.

1. Create [a new issue](https://github.com/serval-uni-lu/tabularbench/issues/new/choose) by selecting the type "Submit a new attack".
2. Fill in the form accordingly.
3. Create a new Pull Request with the dataset definition and files (csv, ...) and associate it with this issue.
4. We will validate that the attack is working correctly, and run the architectures and defenses of the benchmark on it.
5. Once included, the attack will be accessible on the API and on the public leaderboard.

If you find issues with existing attack in the API, please raise a dedicated issue and do not use the form.

Thank you for your contributions.
