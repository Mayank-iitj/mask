"""Settings module that provides access to mask_config."""
import mask_config

# Create a simple config object that wraps mask_config
class Config:
    def __getattr__(self, name):
        return getattr(mask_config, name)

config = Config()
