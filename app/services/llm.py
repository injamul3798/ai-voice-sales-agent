 
import logging
from typing import List, Dict, Any, Optional
import random
try:
    from transformers import pipeline, Conversation
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

SALES_SYSTEM_PROMPT = (
    "You are an AI sales agent for TechEd Academy. "
    "You are selling the 'AI Mastery Bootcamp': a 12-week course for $499 (special offer: $299). "
    "Key benefits: Learn LLMs, Computer Vision, and MLOps; hands-on projects; job placement assistance; certificate upon completion.\n"
    "Follow this conversation flow: Introduction (greet and introduce yourself/company), Qualification (ask 2-3 questions to understand customer needs), Pitch (present the course, customize to user answers), Objection Handling (address price, time, or relevance concerns), Closing (try to schedule a follow-up or get commitment).\n"
    "Always stay on topic, be friendly, and never go off-script.\n"
    "Current stage: {stage}.\n"
    "Conversation so far:\n{history}\n"
    "Your next reply should move the conversation forward in the sales flow."
)

class LLMService:
    """LLM service using HuggingFace Transformers for dynamic, context-aware sales conversations."""
    def __init__(self):
        self.course_info = {
            "name": "AI Mastery Bootcamp",
            "duration": "12 weeks",
            "price": 499,
            "discount_price": 299,
            "features": [
                "Learn LLMs, Computer Vision, and MLOps",
                "Hands-on projects",
                "Job placement assistance",
                "Certificate upon completion"
            ]
        }
        self.hf_pipeline = None
        self.langchain_llm = None
        self.langchain_prompt = None
        if HF_AVAILABLE:
            try:
                self.hf_pipeline = pipeline(
                    "conversational",
                    model="microsoft/DialoGPT-medium",
                    tokenizer="microsoft/DialoGPT-medium"
                )
                logger.info("Loaded HuggingFace conversational pipeline (DialoGPT-medium)")
            except Exception as e:
                logger.error(f"Could not load HuggingFace pipeline: {e}")
                self.hf_pipeline = None
        else:
            logger.warning("Transformers not available, using fallback logic.")
        if LANGCHAIN_AVAILABLE and HF_AVAILABLE:
            try:
                self.langchain_llm = HuggingFacePipeline.from_pipeline(self.hf_pipeline)
                self.langchain_prompt = PromptTemplate(
                    input_variables=["stage", "history"],
                    template=SALES_SYSTEM_PROMPT
                )
                logger.info("LangChain LLM initialized with HuggingFace pipeline.")
            except Exception as e:
                logger.error(f"Could not initialize LangChain LLM: {e}")
                self.langchain_llm = None

    def _get_stage(self, history: List[Dict], customer_info: Dict[str, Any]) -> str:
        # Simple heuristic for stage tracking
        agent_msgs = [m for m in history if m["role"] == "agent"]
        customer_msgs = [m for m in history if m["role"] == "customer"]
        if not agent_msgs:
            return "introduction"
        if len(agent_msgs) < 2:
            return "qualification"
        if len(agent_msgs) < 5:
            return "pitch"
        if any(any(x in m["content"].lower() for x in ["price", "cost", "expensive", "busy", "time"]) for m in customer_msgs):
            return "objection_handling"
        if len(agent_msgs) >= 5:
            return "closing"
        return "pitch"

    def generate_introduction(self, customer_name: str) -> str:
        return f"Hello {customer_name}, I'm your AI sales agent from TechEd Academy. I'm excited to tell you about our AI Mastery Bootcamp. Are you interested in learning more?"

    def generate_response(self, message: str, history: List[Dict], customer_info: Dict[str, Any]) -> tuple[str, bool]:
        stage = self._get_stage(history, customer_info)
        history_str = "\n".join([
            f"Agent: {m['content']}" if m["role"] == "agent" else f"Customer: {m['content']}" for m in history
        ] + [f"Customer: {message}"])
        prompt = SALES_SYSTEM_PROMPT.format(stage=stage, history=history_str)
        enrolled_keywords = ["yes", "enroll", "register", "sign up", "send link", "email", "okay"]
        should_end_call = False
        if stage == "closing":
            last_customer = message.lower()
            if any(k in last_customer for k in enrolled_keywords):
                customer_info['enrolled'] = True
                customer_info['call_ended'] = True
                should_end_call = True
        if customer_info.get('enrolled', False):
            customer_info['call_ended'] = True
            should_end_call = True
            return ("Thank you! I've sent you the enrollment link by email. Looking forward to seeing you in the Bootcamp!", should_end_call)
        # LangChain path
        if self.langchain_llm and self.langchain_prompt:
            try:
                lc_prompt = self.langchain_prompt.format(stage=stage, history=history_str)
                reply = self.langchain_llm(lc_prompt)
                return (reply.strip(), should_end_call)
            except Exception as e:
                logger.error(f"Error with LangChain LLM: {e}")
        # HuggingFace fallback
        if self.hf_pipeline:
            try:
                conv = Conversation(prompt)
                result = self.hf_pipeline(conv)
                reply = result.generated_responses[-1] if hasattr(result, 'generated_responses') else str(result)
                return (reply.strip(), should_end_call)
            except Exception as e:
                logger.error(f"Error with HuggingFace LLM: {e}")
                return ("I'm sorry, I had trouble generating a response. Could you please repeat that?", should_end_call)
        # Fallback: static but context-aware
        if stage == "introduction":
            return (self.generate_introduction(customer_info.get("name", "there")), should_end_call)
        elif stage == "qualification":
            return ("Can you tell me a bit about your technical background and your goals in AI?", should_end_call)
        elif stage == "pitch":
            return ("Based on what you've shared, our AI Mastery Bootcamp is a great fit. It covers LLMs, Computer Vision, and MLOps, with job placement support. Would you like more details?", should_end_call)
        elif stage == "objection_handling":
            return ("I understand your concerns. We offer flexible payment plans and the program is designed for busy professionals. Would you like to discuss further?", should_end_call)
        elif stage == "closing":
            return ("Would you like to schedule a follow-up call or get started with enrollment?", should_end_call)
        return ("Thank you for your message!", should_end_call)

    def is_call_ended(self, customer_info: Dict[str, Any]) -> bool:
        return customer_info.get('call_ended', False)

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Simple message analysis for sentiment and intent."""
        message_lower = message.lower()
        
        # Simple sentiment analysis
        positive_words = ["yes", "interested", "great", "good", "sounds good", "perfect", "love", "excellent"]
        negative_words = ["no", "not interested", "expensive", "busy", "difficult", "hard", "bad", "terrible"]
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            sentiment = {"label": "positive", "score": 0.8}
        elif negative_count > positive_count:
            sentiment = {"label": "negative", "score": 0.2}
        else:
            sentiment = {"label": "neutral", "score": 0.5}
        
        # Simple intent detection
        intent = "general"
        if any(word in message_lower for word in ["yes", "interested", "tell me more"]):
            intent = "interest"
        elif any(word in message_lower for word in ["no", "not interested", "busy"]):
            intent = "rejection"
        elif any(word in message_lower for word in ["price", "cost", "expensive"]):
            intent = "pricing_inquiry"
        elif any(word in message_lower for word in ["time", "schedule", "busy"]):
            intent = "time_inquiry"
        
        # Extract topics
        topics = []
        if any(word in message_lower for word in ["background", "experience", "skills"]):
            topics.append("background")
        if any(word in message_lower for word in ["goals", "career", "job"]):
            topics.append("career_goals")
        if any(word in message_lower for word in ["timeline", "when", "schedule"]):
            topics.append("timeline")
        
        return {
            "sentiment": sentiment,
            "intent": intent,
            "topics": topics
        }

    def _determine_stage(self, history: List[Dict], customer_info: Dict) -> str:
        """Determine the current conversation stage."""
        # If this is the first customer message, it's introduction
        if len(history) <= 1:
            return "introduction"
        
        # Get all customer messages
        customer_messages = [msg for msg in history if msg["role"] == "customer"]
        agent_messages = [msg for msg in history if msg["role"] == "agent"]
        
        # Check if we're in qualification stage
        if "qualification_questions" in customer_info:
            questions_asked = len(customer_info["qualification_questions"])
            if questions_asked < 3:  # We ask 3 qualification questions
                return "qualification"
        
        # Check if we've delivered the pitch
        if "pitch_delivered" in customer_info:
            # Check if customer has shown strong interest or objection
            last_customer_message = ""
            for msg in reversed(customer_messages):
                last_customer_message = msg.get("content", "").lower()
                break
            
            # If customer shows strong interest, move to closing
            if any(word in last_customer_message for word in ["yes", "enroll", "sign up", "register", "ready", "start"]):
                return "closing"
            # If customer shows strong objection, stay in pitch to handle objection
            elif any(word in last_customer_message for word in ["no", "not interested", "expensive", "busy", "later"]):
                return "pitch"
            # If we've had several exchanges in pitch stage, move to closing
            elif len(agent_messages) >= 6:  # After several exchanges
                return "closing"
            else:
                return "pitch"
        
        # If we have profile information, we're ready for pitch
        if "profile" in customer_info:
            return "pitch"
        
        # Check if we should move from introduction to qualification
        # Look for positive indicators in customer messages
        positive_indicators = [
            "yes", "sure", "okay", "interested", "tell me", "learn more", 
            "sounds interesting", "appreciate", "open to", "exploring", 
            "deepen", "skills", "opportunities", "connect", "details",
            "email", "call", "brief", "helpful", "follow up"
        ]
        
        # Check if any customer message shows positive interest
        for msg in customer_messages:
            message_lower = msg.get("content", "").lower()
            if any(word in message_lower for word in positive_indicators):
                return "qualification"
        
        # Default to introduction
        return "introduction"