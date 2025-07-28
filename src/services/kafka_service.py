
import json
from typing import Dict, Any
from src.schemas.requests import ProfanityCheckRequest
from src.services.kafka.producer import KafkaProducerManager
from src.core.config import settings

from src.core.logger import logger


class KafkaIntegrationService:
    def __init__(self):
        self.producer_manager = KafkaProducerManager()
        # Topic names
        self.KAFKA_MODERATION_RESULTS_TOPIC = settings.KAFKA_MODERATION_RESULTS_TOPIC

    def send_moderation_result(self, request_data : any, response_data: Dict[str, Any]):
        """Send moderation result to Kafka topics"""
        if not settings.KAFKA_ENABLED:
            logger.debug("Kafka disabled, skipping message send")
            return

        try:
            # Format and send moderation result                
            success = self.producer_manager.send_message(
                topic=self.KAFKA_MODERATION_RESULTS_TOPIC,
                message= {
                    "request_data": json.loads(request_data) if isinstance(request_data, str) else request_data,
                    "response_data": response_data
                }
            )
            
            if success:
                logger.info("Profanity result sent to Kafka")

        except Exception as e:
            logger.exception("Error sending moderation messages to Kafka:")


    def flush_messages(self):
        """Flush any pending messages"""
        if settings.KAFKA_ENABLED:
            self.producer_manager.flush()

    def shutdown(self):
        """Shutdown Kafka service"""
        logger.info("Shutting down Profanity Kafka service...")
        
        self.executor.shutdown(wait=True)
        self.producer_manager.close()

        logger.info("Profanity Kafka service shutdown complete")