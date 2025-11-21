"""Settings module that re-exports config from mask_config."""
from mask_config import *

# Create a config object for backward compatibility
class Config:
    pass

config = Config()

# Copy all attributes from mask_config to config object
import mask_config as mc
for attr in dir(mc):
    if not attr.startswith('_'):
        setattr(config, attr, getattr(mc, attr))
