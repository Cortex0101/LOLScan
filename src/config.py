"""Configuration management for LOLScan."""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration handler for LOLScan."""

    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {self.config_path}")
        return config or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'model.name', 'training.epochs')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to configuration."""
        return self.get(key)
