"""
Basic unit tests for YOLO Inference Backend refactored code.

These tests verify the core functionality of the refactored modules
without requiring YOLO models or dependencies.
"""

import sys
import os
import tempfile
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_config_singleton(self):
        """Test that Config implements singleton pattern."""
        from config import Config
        
        config1 = Config()
        config2 = Config()
        
        self.assertIs(config1, config2, "Config should be a singleton")
    
    def test_config_validation(self):
        """Test configuration validation."""
        from config import Config
        
        config = Config()
        
        # Valid configuration should pass
        result = config.validate()
        self.assertTrue(result)
    
    def test_config_default_values(self):
        """Test that default configuration values are set."""
        from config import Config
        
        config = Config()
        
        self.assertEqual(config.models_path, './models')
        self.assertEqual(config.conf_thres, 0.25)
        self.assertEqual(config.iou_thres, 0.45)
        self.assertEqual(config.host, '0.0.0.0')
        self.assertEqual(config.port, 8000)
    
    def test_config_string_representation(self):
        """Test string representation of config."""
        from config import Config
        
        config = Config()
        config_str = str(config)
        
        self.assertIn('models_path', config_str)
        self.assertIn('conf_thres', config_str)
        self.assertIn('port', config_str)


class TestLogger(unittest.TestCase):
    """Test logging module."""
    
    def test_setup_logging(self):
        """Test logger setup."""
        from logger import setup_logging, get_logger
        
        logger = setup_logging(log_level='INFO')
        self.assertIsNotNone(logger)
    
    def test_get_logger(self):
        """Test getting logger instance."""
        from logger import get_logger
        
        logger = get_logger('test_module')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_module')


class TestDataModels(unittest.TestCase):
    """Test data models."""
    
    def test_metadata_creation(self):
        """Test Metadata dataclass creation."""
        from utils.dataModel import Metadata
        
        metadata = Metadata(
            training_data="test_data",
            date_trained="2025-01-01",
            maintainers="test_team",
            note="test note"
        )
        
        self.assertEqual(metadata.training_data, "test_data")
        self.assertEqual(metadata.date_trained, "2025-01-01")
    
    def test_models_initialization(self):
        """Test Models class initialization."""
        from utils.dataModel import Models
        
        models = Models(models={})
        self.assertEqual(len(models.models), 0)
    
    def test_models_to_dict(self):
        """Test Models to_dict method."""
        from utils.dataModel import Models
        
        models = Models(models={})
        result = models.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult class."""
    
    def test_detection_result_creation(self):
        """Test DetectionResult can be created and converted to dict."""
        # We need to mock torch for this, so we'll skip if torch not available
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            # Try importing but catch if dependencies not available
            from services.detection_service import DetectionResult
            
            result = DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.95,
                x1=100,
                y1=150,
                x2=300,
                y2=400
            )
            
            result_dict = result.to_dict()
            
            self.assertEqual(result_dict['class_id'], 0)
            self.assertEqual(result_dict['class_name'], 'person')
            self.assertEqual(result_dict['confidence'], 0.95)
            self.assertEqual(result_dict['x1'], 100)
        except ImportError:
            # Skip if torch not available
            self.skipTest("Torch not available, skipping detection result test")


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
