from typing import Protocol, Any

class BaseTracker(Protocol):
    def track(self, *args: Any, **kwargs: Any) -> Any: ...