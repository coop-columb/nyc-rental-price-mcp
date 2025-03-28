import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing preprocessing functions"""
        # Create a simple DataFrame with missing values
        self.data = pd.DataFrame(
            {
                "numeric_feature": [1.0, 2.0, np.nan, 4.0, 5.0],
                "categorical_feature": ["A", "B", None, "A", "C"],
            }
        )

    def test_handle_missing_values(self):
        """Test that missing values can be properly handled"""
        # Test numeric missing value imputation with mean
        numeric_data = self.data["numeric_feature"].copy()
        numeric_data_filled = numeric_data.fillna(numeric_data.mean())

        # Verify no NaN values remain
        self.assertFalse(numeric_data_filled.isna().any())

        # Verify the imputed value equals the mean of non-missing values
        expected_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4
        self.assertEqual(numeric_data_filled[2], expected_mean)

        # Test categorical missing value imputation with most frequent value
        categorical_data = self.data["categorical_feature"].copy()
        most_frequent = categorical_data.mode()[0]
        categorical_data_filled = categorical_data.fillna(most_frequent)

        # Verify no None values remain
        self.assertFalse(categorical_data_filled.isna().any())

        # Verify the imputed value equals the most frequent value
        self.assertEqual(categorical_data_filled[2], "A")

    def test_normalization(self):
        """Test data normalization methods"""
        # Create sample data for normalization
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        # Test StandardScaler (z-score normalization)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Verify mean is close to 0 and std is close to 1
        self.assertTrue(abs(scaled_data.mean()) < 1e-10)
        self.assertAlmostEqual(scaled_data.std(), 1.0)

        # Test MinMaxScaler (min-max normalization)
        min_max_scaler = MinMaxScaler()
        min_max_scaled = min_max_scaler.fit_transform(data)

        # Verify values are in the [0,1] range
        self.assertGreaterEqual(min_max_scaled.min(), 0.0)
        self.assertLessEqual(min_max_scaled.max(), 1.0)

        # Verify the scaling is correct
        self.assertAlmostEqual(min_max_scaled[0][0], 0.0)
        self.assertAlmostEqual(min_max_scaled[-1][0], 1.0)


if __name__ == "__main__":
    unittest.main()
