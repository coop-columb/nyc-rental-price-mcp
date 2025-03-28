import unittest
import tensorflow as tf
import numpy as np

class TestModelBasics(unittest.TestCase):
    """
    Basic tests for model functionality.
    These tests verify that we can create a simple TensorFlow model
    and perform basic operations with it.
    """
    
    def test_create_simple_model(self):
        """Test that we can create a simple TensorFlow model."""
        # Create a very simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Assert that the model has the expected number of layers
        self.assertEqual(len(model.layers), 2)
        
        # Assert that the model has the expected input shape
        self.assertEqual(model.input_shape, (None, 5))
        
        # Assert that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_model_can_predict(self):
        """Test that a simple model can make predictions."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(1)
        ])
        
        # Generate a sample input
        sample_input = np.array([[1.0, 2.0]])
        
        # Make a prediction
        prediction = model.predict(sample_input)
        
        # Check that the prediction has the expected shape
        self.assertEqual(prediction.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()

