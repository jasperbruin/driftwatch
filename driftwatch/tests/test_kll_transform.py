import numpy as np
import pytest
from datasketches import kll_floats_sketch

from utils import kll_transform

###############################################################################
# Basic Functionality Tests
###############################################################################


def test_fit_returns_self():
    """Why important:
      Ensures the fit() method follows standard scikit-like convention of returning self.
    How:
      Create a KLLTransformer, fit it on dummy data, check the returned object is the same instance.
    Expected:
      The return value from fit(...) is the same instance as the transformer.
    """
    X = np.random.rand(10, 3)
    transformer = kll_transform(k=50)
    result = transformer.fit(X)
    assert result is transformer, "fit method should return the instance itself."


def test_fit_populates_sketches():
    """Why important:
      Verifies that after calling fit, we actually get a list of KLL sketches for each dimension.
    How:
      Create a small random dataset, fit the transformer, then check kll_sketches is
      of correct length and has non-empty sketches.
    Expected:
      kll_sketches is a non-empty list, with len == number of dimensions in X.
    """
    X = np.random.rand(10, 5)
    transformer = kll_transform(k=50)
    transformer.fit(X)

    assert transformer.kll_sketches is not None, (
        "kll_sketches should not be None after fit."
    )
    assert len(transformer.kll_sketches) == 5, (
        "Should have one KLL sketch per dimension."
    )
    for sketch in transformer.kll_sketches:
        assert isinstance(sketch, kll_floats_sketch), (
            "Each item in kll_sketches should be a KLL sketch."
        )


def test_transform_basic_functionality():
    """Why important:
      Ensures transform() produces an output of the same shape and that ranks are in [0,1].
    How:
      Fit on random data, then transform a new random dataset of the same shape, check output constraints.
    Expected:
      The shape of output matches input, and all ranks are between 0 and 1 inclusive.
    """
    train_data = np.random.rand(100, 5)
    test_data = np.random.rand(20, 5)

    transformer = kll_transform(k=50)
    transformer.fit(train_data)
    transformed = transformer.transform(test_data)

    assert transformed.shape == test_data.shape, (
        "Output shape should match input shape."
    )
    assert np.all(transformed >= 0.0) and np.all(transformed <= 1.0), (
        "All transformed values should lie between 0 and 1."
    )


###############################################################################
# Edge Case Tests
###############################################################################


@pytest.mark.parametrize(
    "X, expected_dim",
    [
        (np.array([[1, 2, 3]]), 3),  # single row
        (np.random.rand(1, 5), 5),  # single sample, multiple dims
        (np.random.rand(50, 1), 1),  # single dimension, multiple samples
    ],
)
def test_fit_with_various_shapes(X, expected_dim):
    """Why important:
      Covers scenarios where X might have only 1 row or 1 column, etc.
    How:
      Provide different shapes to fit, then ensure sketches are created and have correct dimension count.
    Expected:
      The number of KLL sketches matches the number of columns in X.
    """
    transformer = kll_transform(k=50)
    transformer.fit(X)

    assert len(transformer.kll_sketches) == expected_dim, (
        f"Number of sketches should be {expected_dim} for input shape {X.shape}."
    )


def test_transform_same_data_after_fit():
    """Why important:
      Sometimes we want to verify the transform is consistent if we transform the same data used in fit.
    How:
      Fit on a random dataset, then transform that same dataset and ensure the shape is consistent
      and the ranks do not produce out-of-range values.
    Expected:
      Valid transformation (all rank values in [0,1]).
    """
    X = np.random.rand(10, 2)
    transformer = kll_transform(k=10)
    transformer.fit(X)
    transformed = transformer.transform(X)

    assert transformed.shape == X.shape, (
        "Transformed data should maintain the same shape."
    )
    assert np.all(transformed >= 0.0) and np.all(transformed <= 1.0), (
        "Rank values should be within [0,1] for all points."
    )


def test_transform_empty_array():
    """Why important:
      Edge case with zero samples. Some transformations might fail on empty arrays.
    How:
      Fit on a normal dataset, then call transform on an empty array with correct dimensionality.
    Expected:
      Transform should return an empty array of matching shape (0, embedding_dim), no exception.
    """
    X = np.random.rand(10, 3)
    transformer = kll_transform(k=20)
    transformer.fit(X)

    empty_array = np.empty((0, 3))
    transformed = transformer.transform(empty_array)

    assert transformed.shape == (0, 3), (
        "Should return an empty array with same number of dimensions."
    )


###############################################################################
# Error Handling Tests
###############################################################################


def test_transform_before_fit_raises_error():
    """Why important:
      If transform is called before fit, kll_sketches is None, so we expect an error.
    How:
      Create transformer, do not call fit, then call transform. Check for an AttributeError or custom error.
    Expected:
      The transform method should raise an error due to uninitialized sketches.
    """
    transformer = kll_transform(k=50)
    X = np.random.rand(5, 2)

    with pytest.raises(AttributeError):
        _ = transformer.transform(X)


def test_inconsistent_dimension_raises_error():
    """Why important:
      Ensures that if transform is called with data that has different dimensionality than what was used in fit,
      it raises an error (or handles it gracefully).
    How:
      Fit on shape (num_samples, 3), then transform shape (num_samples, 4), check for an error.
    Expected:
      The transform method should raise an error regarding inconsistent dimension.
    """
    X_fit = np.random.rand(10, 3)
    X_transform = np.random.rand(5, 4)  # different dimension

    transformer = kll_transform(k=50)
    transformer.fit(X_fit)

    with pytest.raises(ValueError):
        _ = transformer.transform(X_transform)


def test_invalid_k_value():
    """Why important:
      k should be a positive integer; negative or zero doesn't make sense for KLL sketches.
    How:
      Try initializing the transformer with an invalid k.
    Expected:
      Depending on implementation, it should raise an error (or we test that it handles gracefully).
    """
    with pytest.raises(ValueError):
        # This assumes the __init__ method has a check for k <= 0. If it doesn't,
        # you might add such a check or simply remove this test if it's not needed.
        _ = kll_transform(k=0)


###############################################################################
# Parameterized Tests for Different k Values
###############################################################################


@pytest.mark.parametrize("k_value", [10, 50, 100])
def test_various_k_values(k_value):
    """Why important:
      Different k sizes affect sketch accuracy and memory usage; we want to ensure it still runs for typical k values.
    How:
      Fit a small random dataset with different k values.
    Expected:
      No error occurs, transformation completes successfully, and output shape is correct.
    """
    X_train = np.random.rand(20, 3)
    X_test = np.random.rand(5, 3)

    transformer = kll_transform(k=k_value)
    transformer.fit(X_train)
    transformed = transformer.transform(X_test)

    assert transformed.shape == (5, 3), (
        f"Output shape must match (5, 3) for k={k_value}."
    )
    assert np.all(transformed >= 0.0) and np.all(transformed <= 1.0), (
        f"For k={k_value}, ranks should be between 0 and 1."
    )


###############################################################################
# Optional: Setup/Teardown Fixtures
###############################################################################
@pytest.fixture
def simple_dataset():
    """Example of a setup fixture if needed for multiple tests.
    Returns a small random dataset.
    """
    return np.random.rand(10, 3)


def test_using_fixture(simple_dataset):
    """Why important:
      Demonstrates usage of a fixture (if stateful setup is ever needed).
    How:
      Use the simple_dataset fixture to fit/transform, verifying it works as expected.
    Expected:
      The transformation yields ranks in [0,1].
    """
    transformer = kll_transform(k=15)
    transformer.fit(simple_dataset)
    transformed = transformer.transform(simple_dataset)

    assert transformed.shape == simple_dataset.shape
    assert np.all(transformed >= 0.0) and np.all(transformed <= 1.0)


if __name__ == "__main__":
    pytest.main()
