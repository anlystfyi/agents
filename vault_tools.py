from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional

import requests
from phi.tools import Toolkit
from phi.utils.log import logger


@dataclass
class VaultConfig:
    """Configuration for Vault API connection"""
    base_url: str = "https://vault-api.anlyst.ai"
    token: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        self.token = self.token or getenv("VAULT_API_KEY")
        if not self.token:
            logger.error("No Vault API token provided")

class VaultAPITools(Toolkit):
    """Toolkit for interacting with Vault API endpoints"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_entries: bool = True,
        enable_history: bool = True,
    ) -> None:
        """
        Initialize Vault API toolkit with selective tool registration
        
        Args:
            api_key: Optional API key (falls back to env var)
            enable_entries: Enable entries related tools
            enable_history: Enable history related tools
        """
        super().__init__(name="vault_api_tools")
        self.config = VaultConfig(token=api_key)
        
        # Selective tool registration
        if enable_entries:
            self.register(self.get_entries_by_key)
            self.register(self.get_current_entry)
        if enable_history:
            self.register(self.get_entry_history)
    
    def get_entries_by_key(self, key: str, timeout: Optional[int] = 30) -> Dict[str, Any]:
        """
        Get entries by key from Vault
        
        Args:
            key: Key to fetch (e.g., 'sleep', 'recovery')
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing entry data or error message
        """
        if self.config.token is None:
            return {"error": "No API token provided"}
        
        try:
            logger.debug(f"Fetching entries for key: {key}")
            response = requests.get(
                f"{self.config.base_url}/entries/by_key/{key}",
                headers={"Authorization": f"Bearer {self.config.token}"},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching entries: {str(e)}")
            return {"error": str(e)}
    
    def get_current_entry(self, key: str) -> Dict[str, Any]:
        """
        Get current entry for a key
        
        Args:
            key: Key to fetch current entry for
            
        Returns:
            Dict containing current entry or error message
        """
        try:
            logger.debug(f"Fetching current entry for key: {key}")
            data = self.get_entries_by_key(key)
            if "error" in data:
                return data
            return data.get("current", {})
        except Exception as e:
            logger.error(f"Error fetching current entry: {str(e)}")
            return {"error": str(e)}
    
    def get_entry_history(
        self, 
        key: str,
        limit: Optional[int] = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical entries for a key
        
        Args:
            key: Key to fetch history for
            limit: Maximum number of historical entries to return
            
        Returns:
            List of historical entries or error message
        """
        try:
            logger.debug(f"Fetching history for key: {key}")
            data = self.get_entries_by_key(key)
            if "error" in data:
                return [data]
            history = data.get("history", [])
            return history[:limit] if limit else history
        except Exception as e:
            logger.error(f"Error fetching history: {str(e)}")
            return [{"error": str(e)}]