"""Secure Credential Loader.

Loads Kraken API credentials from environment variables or ~/.kraken_creds file.
File permissions must be 600 for security.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SecureCreds:
    """
    Secure credential manager for Kraken API.
    
    Loads credentials from:
    1. Environment variables (KRAKEN_API_KEY, KRAKEN_API_SECRET)
    2. ~/.kraken_creds file (format: key=VALUE, secret=VALUE)
    
    The credentials file must have 600 permissions for security.
    """
    
    ENV_API_KEY = "KRAKEN_API_KEY"
    ENV_API_SECRET = "KRAKEN_API_SECRET"
    CREDS_FILE = Path.home() / ".kraken_creds"
    
    def __init__(self):
        """Initialize and load credentials."""
        self._api_key: Optional[str] = None
        self._api_secret: Optional[str] = None
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from environment or file."""
        # Try environment variables first
        env_key = os.environ.get(self.ENV_API_KEY)
        env_secret = os.environ.get(self.ENV_API_SECRET)
        
        if env_key and env_secret:
            logger.info("Credentials loaded from environment variables")
            self._api_key = env_key
            self._api_secret = env_secret
            return
        
        # Try credentials file
        creds_file = self.CREDS_FILE
        if creds_file.exists():
            self._validate_file_permissions(creds_file)
            self._parse_creds_file(creds_file)
        else:
            logger.debug(f"Credentials file not found: {creds_file}")
    
    def _validate_file_permissions(self, path: Path) -> None:
        """Validate that credentials file has secure permissions (600)."""
        try:
            mode = path.stat().st_mode & 0o777
            if mode != 0o600:
                logger.warning(
                    f"Insecure file permissions on {path}: {oct(mode)}. "
                    f"Should be 600. Run: chmod 600 {path}"
                )
        except OSError as e:
            logger.error(f"Could not check file permissions: {e}")
    
    def _parse_creds_file(self, path: Path) -> None:
        """Parse credentials from file (key=value format)."""
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        # Support multiple formats:
                        # key=value, secret=value
                        # kraken_api_key=value, kraken_private_key=value
                        if key == 'key' or key == 'kraken_api_key':
                            self._api_key = value
                        elif key == 'secret' or key == 'kraken_private_key':
                            self._api_secret = value
            
            if self._api_key and self._api_secret:
                logger.info("Credentials loaded from file")
            else:
                logger.warning("Incomplete credentials in file")
        except OSError as e:
            logger.error(f"Error reading credentials file: {e}")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get API key."""
        return self._api_key
    
    @property
    def api_secret(self) -> Optional[str]:
        """Get API secret."""
        return self._api_secret
    
    @property
    def is_available(self) -> bool:
        """Check if credentials are available."""
        return bool(self._api_key and self._api_secret)
    
    def clear(self) -> None:
        """Clear loaded credentials."""
        self._api_key = None
        self._api_secret = None
        logger.debug("Credentials cleared")


if __name__ == "__main__":
    creds = SecureCreds()
    print(f"API Key: {'*' * len(creds.api_key) if creds.api_key else 'Not set'}")
    print(f"API Secret: {'*' * len(creds.api_secret) if creds.api_secret else 'Not set'}")
    print(f"Available: {creds.is_available}")
