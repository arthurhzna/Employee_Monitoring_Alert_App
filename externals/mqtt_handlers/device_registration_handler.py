from externals.mqtt_handlers.base_handler import MQTTChannelHandler
from core.state_manager import StateManager
from datetime import datetime
from typing import Dict, Any

class DeviceRegistrationHandler(MQTTChannelHandler):
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    def can_handle(self, topic: str, action: str) -> bool:
        return (
            "alert/subscribe" in topic and 
            action in ["device_registered", "device_reset"]
        )
    
    def handle(self, topic: str, payload: Dict[str, Any]) -> None:
        action = payload.get("action")
        
        if action == "device_registered":
            self._handle_registered()
        elif action == "device_reset":
            self._handle_reset()
    
    def _handle_registered(self) -> None:
        
        self.state_manager.update_device_registration(
            is_registered=True
        )
        

    def _handle_reset(self) -> None:
        
        self.state_manager.update_device_registration(
            is_registered=False
        )
        