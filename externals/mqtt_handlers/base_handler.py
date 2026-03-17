# externals/redis_handlers/base_handler.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class MQTTChannelHandler(ABC):
    """Base class untuk Redis channel handlers"""
    
    @abstractmethod
    def can_handle(self, channel: str, message_type: str = None) -> bool:
        """Check apakah handler ini bisa handle channel ini"""
        pass
    
    @abstractmethod
    def handle(self, channel: str, message: str) -> None:
        """Handle message dari Redis channel"""
        pass