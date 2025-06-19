"""
Configuration settings for AI Voice Sales Agent
"""

class Settings:
    # Application settings
    APP_NAME: str = "AI Voice Sales Agent"
    DEBUG: bool = True
    
    # Course information
    COURSE_NAME: str = "AI Mastery Bootcamp"
    COURSE_DURATION: str = "12 weeks"
    COURSE_PRICE: float = 499.0
    COURSE_DISCOUNT_PRICE: float = 299.0
    
    # Conversation settings
    MAX_CONVERSATION_TURNS: int = 20
    TEMPERATURE: float = 0.7

settings = Settings() 