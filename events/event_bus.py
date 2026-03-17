# events/event_bus.py
from typing import Callable, Dict, List
from threading import Lock
from events.base_event import BaseEvent

class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[BaseEvent], None]]] = {}
        self._lock = Lock()

    def subscribe(self, event_type: str, handler: Callable[[BaseEvent], None]) -> None:
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[BaseEvent], None]) -> None:
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    h for h in self._subscribers[event_type] if h is not handler
                ]

    def publish(self, event: BaseEvent) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in handler {handler.__name__}: {e}")

    def get_subscribers(self, event_type: str) -> int:
        with self._lock:
            return len(self._subscribers.get(event_type, []))