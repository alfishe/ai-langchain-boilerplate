from abc import ABC, abstractmethod
from typing import Callable, Any

class PubSubBroker(ABC):
    """Interface for pub/sub broker (Redis, Kafka, MQTT)"""
    
    @abstractmethod
    def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic"""
        pass
    
    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to a topic with a callback function"""
        pass
    
    @abstractmethod
    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic"""
        pass 