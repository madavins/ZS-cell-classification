import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML
        
    Returns:
        Dictionary containing validated configuration
    """
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file {config_path}: {e}")
        
    return config