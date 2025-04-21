import numpy as np
import pytest

from utils import kll_transform


@pytest.mark.parametrize("invalid_k", [0, -1, 3.5, "hello", 7, 65536])
def test_invalid_init(invalid_k):
    """Reasoning:
      k must be an integer within the valid range [8, 65535].
      Values outside this range or non-integer values should raise ValueError.
    Methodology:
      Attempt to instantiate KLLTransformer with invalid values.
    Expected Output:
      ValueError raised for invalid k.
    """
    with pytest.raises(ValueError):
        _ = kll_transform(k=invalid_k)


@pytest.mark.parametrize("valid_k", [8, 50, 100, 65535])
def test_valid_init(valid_k):
    """Reasoning:
      k should be allowed within the range [8, 65535].
    Methodology:
      Initialize KLLTransformer with valid k values.
    Expected Output:
      No error raised; transformer is created successfully.
    """
    transformer = kll_transform(k=valid_k)
    assert transformer.k == valid_k, (
        f"KLLTransformer should store the correct value of k={valid_k}."
    )


###############################################################################
# 2. Fitting Tests
###############################################################################
def test_fit_valid_data():
    """Reasoning:
      Fitting on a valid 2D NumPy array should succeed, creating a KLL sketch per column.
    Methodology:
      Use a random array shape (10, 3). Call fit() and verify kll_sketches count equals embedding_dim.
    Expected Output:
      No error, len(kll_sketches) == 3.
    """
    X = np.random.rand(10, 3)
    transformer = kll_transform(k=20)
    transformer.fit(X)
    assert transformer.kll_sketches is not None, (
        "kll_sketches should be initialized after fit."
    )
    assert len(transformer.kll_sketches) == 3, (
        "There should be one KLL sketch per dimension."
    )


def test_fit_non_numpy_input():
    """Reasoning:
      X must be a NumPy array. Passing a list or other type should raise TypeError.
    Methodology:
      Pass a list of lists to fit().
    Expected Output:
      TypeError.
    """
    X_list = [[1, 2, 3], [4, 5, 6]]
    transformer = kll_transform(k=10)
    with pytest.raises(TypeError):
        transformer.fit(X_list)


def test_fit_not_2d():
    """Reasoning:
      X must be a 2D array. 1D or higher-dim arrays are invalid.
    Methodology:
      Pass a 1D array of shape (10,) or a 3D array of shape (2,2,2) to fit().
    Expected Output:
      ValueError for both cases.
    """
    transformer = kll_transform(k=10)
    # 1D array
    X_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        transformer.fit(X_1d)

    # 3D array
    X_3d = np.random.rand(2, 2, 2)
    with pytest.raises(ValueError):
        transformer.fit(X_3d)


def test_fit_zero_columns():
    """Reasoning:
      Even if we have multiple samples, zero columns are not valid (shape=(10, 0)).
    Methodology:
      Construct an array of shape (10, 0) and fit().
    Expected Output:
      ValueError indicating no columns.
    """
    transformer = kll_transform(k=10)
    X_zero_cols = np.empty((10, 0))
    with pytest.raises(ValueError):
        transformer.fit(X_zero_cols)


###############################################################################
# 3. Transform Tests
###############################################################################
def test_transform_basic():
    """Reasoning:
      Transforming data after fit should produce an output of the same shape, with ranks in [0,1].
    Methodology:
      Fit on some random data, then transform new random data of the same shape.
    Expected Output:
      No error; output shape == input shape; all values in [0,1].
    """
    train_data = np.random.rand(10, 3)
    test_data = np.random.rand(5, 3)

    transformer = kll_transform(k=30)
    transformer.fit(train_data)
    result = transformer.transform(test_data)

    assert result.shape == test_data.shape, "Output should match input shape."
    assert np.all((result >= 0) & (result <= 1)), "All ranks should be in [0,1]."


def test_transform_before_fit():
    """Reasoning:
      transform() must not be called before fit(). Should raise AttributeError if kll_sketches is None.
    Methodology:
      Create KLLTransformer, call transform without fit.
    Expected Output:
      AttributeError raised.
    """
    transformer = kll_transform(k=10)
    X = np.random.rand(5, 2)
    with pytest.raises(AttributeError):
        transformer.transform(X)


def test_transform_mismatched_dimensions():
    """Reasoning:
      Transform data must have the same number of dimensions as was fitted.
    Methodology:
      Fit on shape (10,3), then transform shape (5,4).
    Expected Output:
      ValueError raised for mismatch in embedding_dim.
    """
    X_fit = np.random.rand(10, 3)
    X_transform = np.random.rand(5, 4)

    transformer = kll_transform(k=10)
    transformer.fit(X_fit)

    with pytest.raises(ValueError):
        transformer.transform(X_transform)


###############################################################################
# 4. Parameterized Tests
###############################################################################
@pytest.mark.parametrize(
    "X_data, shape_description",
    [
        (np.random.rand(1, 3), "single row"),
        (np.random.rand(10, 1), "single column"),
        (np.random.rand(10, 5), "typical multi-dim"),
    ],
)
def test_fit_transform_varied_shapes(X_data, shape_description):
    """Reasoning:
      Verify the KLLTransformer can handle a variety of shapes (e.g., single row, single column, multiple columns).
    Methodology:
      Fit on X_data, transform the same X_data to check that shape is preserved and ranks are valid.
    Expected Output:
      No error. Output shape matches input, all values are in [0,1].
    """
    transformer = kll_transform(k=20)
    transformer.fit(X_data)
    transformed = transformer.transform(X_data)

    assert transformed.shape == X_data.shape, (
        f"For {shape_description}, output shape should match the input shape."
    )
    assert np.all((transformed >= 0) & (transformed <= 1)), (
        f"For {shape_description}, ranks should be within [0,1]."
    )


@pytest.mark.parametrize("k_value", [5, 50, 100])
def test_transform_different_k(k_value):
    """Reasoning:
      Different k values can affect memory and accuracy, but should not break functionality.
    Methodology:
      Fit and transform a small random dataset with different k values.
    Expected Output:
      No error, rank outputs in [0,1].
    """
    X_train = np.random.rand(10, 3)
    X_test = np.random.rand(5, 3)

    transformer = kll_transform(k=k_value)
    transformer.fit(X_train)
    transformed = transformer.transform(X_test)

    assert transformed.shape == (5, 3), (
        f"Output shape must match input shape for k={k_value}."
    )
    assert np.all((transformed >= 0) & (transformed <= 1)), (
        f"All ranks should be in [0,1] for k={k_value}."
    )


if __name__ == "__main__":
    pytest.main(["-v"])
