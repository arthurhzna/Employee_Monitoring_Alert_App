# externals/mqtt_client.py
import paho.mqtt.client as mqtt
import json
import threading
from typing import Optional, List, Dict, Any
from events.base_event import BaseEvent
from externals.mqtt_handlers.base_handler import MQTTChannelHandler
import core.config as config

class MQTTClient:
    def __init__(
        self,
    ) -> None:
        self.broker = config.Config.mqtt.broker
        self.port = config.Config.mqtt.port
        self.client_id = f"cv_client_{config.Config.device_id}"
        self.username = config.Config.mqtt.user
        self.password = config.Config.mqtt.password
        
        self.client: Optional[mqtt.Client] = None
        self._connected = False
        self._handlers: List[MQTTChannelHandler] = []  
        self._subscribed_topics: List[tuple[str, int]] = [] 
    
    def connect(self) -> bool:
        try:
            self.client = mqtt.Client(client_id=self.client_id)
            
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            # Setup callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()  # Start background thread untuk consume
            return True
        except Exception as e:
            print(f"Failed to connect to MQTT: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"Connected to MQTT broker: {self.broker}:{self.port}")
            
            for topic, qos in self._subscribed_topics:
                try:
                    client.subscribe(topic, qos)
                    print(f"Resubscribed to MQTT topic: {topic} (QoS: {qos})")
                except Exception as e:
                    print(f"Failed to resubscribe to {topic}: {e}")
        else:
            print(f"Failed to connect to MQTT, return code: {rc}")
            self._connected = False
    
    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload_str = msg.payload.decode('utf-8')
        
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            print(f"Invalid JSON payload from topic {topic}: {payload_str}")
            return
        
        action = payload.get("action", "")
        
        handler_found = False
        for handler in self._handlers:
            if handler.can_handle(topic, action):
                try:
                    handler.handle(topic, payload)
                    handler_found = True
                    break
                except Exception as e:
                    print(f"Error in handler {handler.__class__.__name__}: {e}")
        
        if not handler_found:
            print(f"No handler found for topic={topic}, action={action}")
    
    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"Unexpected MQTT disconnection (rc={rc})")
        else:
            print(f"Disconnected from MQTT broker")
    
    def register_handler(self, handler: MQTTChannelHandler) -> None:
        self._handlers.append(handler)
        print(f"Registered MQTT handler: {handler.__class__.__name__}")
    
    def subscribe(self, topic: str, qos: int = 1) -> None:
        topic_qos = (topic, qos)
        if topic_qos not in self._subscribed_topics:
            self._subscribed_topics.append(topic_qos)
        
        if self.client and self._connected:
            try:
                self.client.subscribe(topic, qos)
                print(f"Subscribed to MQTT topic: {topic} (QoS: {qos})")
            except Exception as e:
                print(f"Failed to subscribe to {topic}: {e}")
        else:
            print(f"Queued subscription to {topic} (will subscribe when connected)")
    
    def publish(self, topic: str, payload: str, qos: int = 1) -> bool:
        if not self._connected or not self.client:
            print(f"Warning: Cannot publish to {topic}, not connected")
            return False
        try:
            result = self.client.publish(topic, payload, qos)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"MQTT publish error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            print("MQTT client disconnected")
    
    def is_connected(self) -> bool:
        return self._connected