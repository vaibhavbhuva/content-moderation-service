import threading
from typing import Dict, Any, Optional, Callable
from kafka import KafkaProducer
from src.core.config import settings
from src.core.logger import logger
from src.utils.kafka_utils import MessageSerializer

class KafkaProducerManager:
    def __init__(self):
        self._producer = None
        self._lock = threading.Lock()
        self.bootstrap_servers = settings.KAFKA_BOOTSTRAP_SERVERS

    def get_producer(self) -> KafkaProducer:
        """Get or create Kafka producer instance (thread-safe)"""
        if self._producer is None:
            with self._lock:
                if self._producer is None:
                    self._producer = KafkaProducer(
                        bootstrap_servers=self.bootstrap_servers,
                        value_serializer=MessageSerializer.serialize_value,
                        key_serializer=MessageSerializer.serialize_key,
                        retries=settings.KAFKA_RETRIES,
                        retry_backoff_ms=settings.KAFKA_RETRY_BACKOFF_MS,
                        # acks=settings.KAFKA_ACKS,
                        # batch_size=settings.KAFKA_BATCH_SIZE,
                        # linger_ms=settings.KAFKA_LINGER_MS,
                        # buffer_memory=settings.KAFKA_BUFFER_MEMORY,
                        # max_request_size=settings.KAFKA_MAX_REQUEST_SIZE
                    )
                    logger.info("Kafka producer initialized")
        return self._producer

    def send_message(self, 
                    topic: str, 
                    message: Dict[str, Any], 
                    success_callback: Optional[Callable] = None,
                    error_callback: Optional[Callable] = None) -> bool:
        """Send message to Kafka topic"""
        try:
            logger.debug(f"Kafka producer message received topic : {topic}: {message} ")
            producer = self.get_producer()
            future = producer.send(topic, value=message)
            
            # Add callbacks
            if success_callback:
                future.add_callback(success_callback)
            else:
                future.add_callback(self._default_success_callback)
                
            if error_callback:
                future.add_errback(error_callback)
            else:
                future.add_errback(self._default_error_callback)

            logger.debug(f"Message queued for topic '{topic}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {str(e)}")
            if error_callback:
                error_callback(e)
            return False

    def send_message_sync(self, topic: str, message: Dict[str, Any], key: Optional[str] = None, timeout: int = 10) -> bool:
        """Send message synchronously with timeout"""
        try:
            producer = self.get_producer()
            future = producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=timeout)
            
            logger.info(f"Message sent to topic '{record_metadata.topic}' "
                       f"partition {record_metadata.partition} offset {record_metadata.offset}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message synchronously: {str(e)}")
            return False

    def flush(self):
        """Flush any pending messages"""
        if self._producer:
            self._producer.flush()

    def close(self):
        """Close producer connection"""
        if self._producer:
            self._producer.close()
            self._producer = None
            logger.info("Kafka producer closed")

    def _default_success_callback(self, record_metadata):
        """Default success callback"""
        logger.info(f"Message sent successfully to topic '{record_metadata.topic}' "
                   f"partition {record_metadata.partition} offset {record_metadata.offset}")

    def _default_error_callback(self, exception):
        """Default error callback"""
        logger.error(f"Failed to send message: {str(exception)}")