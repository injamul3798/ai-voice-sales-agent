"""
Conversation Model for AI Voice Sales Agent
Simplified conversation management system.
"""

from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Conversation:
    """Simplified conversation management for sales calls."""

    def __init__(self, customer_name: str, phone_number: Optional[str] = None, customer_info: Optional[Dict[str, Any]] = None):
        """Initialize a new conversation."""
        self.call_id = str(uuid.uuid4())
        self.customer_name = customer_name
        self.phone_number = phone_number
        self.customer_info = customer_info or {}
        self.state = "introduction"
        self.messages: List[Dict] = []
        self.qualification_answers: Dict[str, str] = {}
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        logger.info(f"New conversation started: {self.call_id}")

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update customer info based on message content
        if role == "customer":
            self._update_customer_info(content)

    def _update_customer_info(self, content: str):
        """Update customer info based on message content."""
        content_lower = content.lower()
        
        # Track qualification responses
        if "background" in content_lower or "experience" in content_lower:
            self.customer_info["has_tech_background"] = any(word in content_lower for word in ["python", "programming", "developer", "engineer", "experience"])
        
        if "goals" in content_lower or "career" in content_lower:
            self.customer_info["career_focused"] = any(word in content_lower for word in ["career", "job", "promotion", "skills", "learn"])
        
        if "time" in content_lower or "schedule" in content_lower:
            self.customer_info["time_constrained"] = any(word in content_lower for word in ["busy", "flexible", "weekend", "evening", "hours"])

    def end_call(self):
        """End the conversation."""
        self.end_time = datetime.now()
        self.state = "closing"
        logger.info(f"Call ended: {self.call_id}")

    def get_duration(self) -> float:
        """Get call duration in seconds."""
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary."""
        customer_messages = [msg for msg in self.messages if msg["role"] == "customer"]
        agent_messages = [msg for msg in self.messages if msg["role"] == "agent"]
        
        return {
            "call_id": self.call_id,
            "customer_name": self.customer_name,
            "duration": self.get_duration(),
            "total_messages": len(self.messages),
            "customer_messages": len(customer_messages),
            "agent_messages": len(agent_messages),
            "state": self.state,
            "qualification_complete": len(self.qualification_answers) >= 3,
            "sentiment_summary": self._get_sentiment_summary(),
            "insights": {
                "qualification_answers": self.qualification_answers,
                "conversation_flow": self._get_conversation_flow()
            }
        }

    def _get_sentiment_summary(self) -> Dict:
        """Calculate simple sentiment summary."""
        if not self.messages:
            return {"label": "neutral", "score": 0.5}
        
        # Simple sentiment calculation
        positive_words = ["yes", "interested", "great", "good", "perfect", "love", "excellent"]
        negative_words = ["no", "not interested", "expensive", "busy", "difficult", "bad"]
        
        total_sentiment = 0
        message_count = 0
        
        for msg in self.messages:
            if msg["role"] == "customer":
                content = msg["content"].lower()
                positive_count = sum(1 for word in positive_words if word in content)
                negative_count = sum(1 for word in negative_words if word in content)
                
                if positive_count > negative_count:
                    total_sentiment += 0.8
                elif negative_count > positive_count:
                    total_sentiment += 0.2
                else:
                    total_sentiment += 0.5
                
                message_count += 1
        
        if message_count == 0:
            return {"label": "neutral", "score": 0.5}
        
        avg_sentiment = total_sentiment / message_count
        label = "positive" if avg_sentiment > 0.6 else "negative" if avg_sentiment < 0.4 else "neutral"
        
        return {"label": label, "score": avg_sentiment}

    def _get_conversation_flow(self) -> List[str]:
        """Get conversation flow stages."""
        stages = []
        for msg in self.messages:
            if msg["role"] == "agent":
                content = msg["content"].lower()
                if "background" in content:
                    stages.append("qualification")
                elif "goals" in content:
                    stages.append("qualification")
                elif "timeline" in content:
                    stages.append("qualification")
                elif "bootcamp" in content and "perfect" in content:
                    stages.append("pitch")
                elif "price" in content or "cost" in content:
                    stages.append("objection_handling")
                elif "enrollment" in content or "application" in content:
                    stages.append("closing")
        
        return stages

    def to_dict(self) -> Dict:
        """Convert conversation to dictionary."""
        return {
            "call_id": self.call_id,
            "customer_name": self.customer_name,
            "phone_number": self.phone_number,
            "state": self.state,
            "messages": self.messages,
            "qualification_answers": self.qualification_answers,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        """Create conversation from dictionary."""
        conversation = cls(
            customer_name=data["customer_name"],
            phone_number=data.get("phone_number"),
            customer_info=data.get("customer_info")
        )
        conversation.call_id = data["call_id"]
        conversation.state = data["state"]
        conversation.messages = data["messages"]
        conversation.qualification_answers = data["qualification_answers"]
        conversation.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            conversation.end_time = datetime.fromisoformat(data["end_time"])
        
        return conversation 