class NotSameFormat(Exception):
    pass


def min_max_scaled(a):
    tol = 0.00001
    return (a.min() >= (0 - tol)) and (a.max() <= (1 + tol))


def check_same_format(ref, tested):

    if ref.shape[1] != tested.shape[1]:
        raise NotSameFormat(
            f"Reference has shape 1 {ref.shape[1]}, while tested has shape {tested.shape[1]}"
        )

    if min_max_scaled(ref):
        # Suppose we are in scaled mode.
        if not min_max_scaled(tested):
            raise NotSameFormat(
                f"Reference is min max scaled but tested is not."
            )

    if not min_max_scaled(ref):
        if min_max_scaled(tested):
            NotSameFormat(f"Reference is NOT min max scaled but tested is.")
