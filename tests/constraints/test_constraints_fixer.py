import numpy as np

from tabularbench.constraints.constraints_fixer import ConstraintsFixer
from tabularbench.constraints.relation_constraint import Feature


class TestConstraintsFixer:
    def test_fix(self):

        x = np.arange(9).reshape(3, 3)

        g1 = Feature(0) == Feature(1) + Feature(2)

        constraints_fixer = ConstraintsFixer(
            guard_constraints=[g1],
            fix_constraints=[g1],
        )

        x_fixed = constraints_fixer.fix(x)

        x_expected = np.array([[3, 1, 2], [9, 4, 5], [15, 7, 8]])

        assert np.equal(x_fixed, x_expected).all()

    def test_no_side_effect(self):
        x = np.arange(9).reshape(3, 3)

        g1 = Feature(0) == Feature(1) + Feature(2)

        constraints_fixer = ConstraintsFixer(
            guard_constraints=[g1],
            fix_constraints=[g1],
        )

        x_fixed = constraints_fixer.fix(x)

        x_expected_original = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        x_expected = np.array([[3, 1, 2], [9, 4, 5], [15, 7, 8]])

        assert np.equal(x, x_expected_original).all()
        assert np.equal(x_fixed, x_expected).all()
