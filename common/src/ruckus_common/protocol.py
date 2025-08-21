"""Wire protocol for RUCKUS communication."""

from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid


class MessageType(str, Enum):
    """Types of protocol messages."""
    # Agent -> Server
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    JOB_UPDATE = "job_update"
    JOB_COMPLETE = "job_complete"
    ERROR = "error"

    # Server -> Agent
    JOB_REQUEST = "job_request"
    JOB_CANCEL = "job_cancel"
    SHUTDOWN = "shutdown"

    # Bidirectional
    PING = "ping"
    PONG = "pong"
    ACK = "ack"


class Message(BaseModel):
    """Base message format for all communications."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender_id: str
    recipient_id: Optional[str] = None
    correlation_id: Optional[str] = None  # For request-response correlation
    payload: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class Request(Message):
    """Request message requiring a response."""
    requires_response: bool = True
    timeout_seconds: int = Field(default=30, gt=0)


class Response(Message):
    """Response to a request."""
    success: bool
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    @validator("correlation_id")
    def correlation_required(cls, v):
        if not v:
            raise ValueError("correlation_id is required for responses")
        return v


class ErrorResponse(Response):
    """Error response message."""
    success: bool = False
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    retry_able: bool = False


# Specific Protocol Messages
class RegisterRequest(Request):
    """Agent registration request."""
    message_type: MessageType = MessageType.REGISTER

    class Config:
        use_enum_values = True


class RegisterResponse(Response):
    """Response to registration."""
    agent_token: Optional[str] = None  # For authenticated communication
    assigned_id: Optional[str] = None  # Server-assigned agent ID
    config_overrides: Optional[Dict[str, Any]] = None


class HeartbeatMessage(Message):
    """Agent heartbeat."""
    message_type: MessageType = MessageType.HEARTBEAT
    agent_status: str  # "idle", "busy", "error"
    current_jobs: List[str] = Field(default_factory=list)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class JobRequestMessage(Request):
    """Server requesting agent to execute a job."""
    message_type: MessageType = MessageType.JOB_REQUEST
    job_id: str
    priority: int = 0

    class Config:
        use_enum_values = True


class JobUpdateMessage(Message):
    """Agent updating job progress."""
    message_type: MessageType = MessageType.JOB_UPDATE
    job_id: str

    class Config:
        use_enum_values = True


class JobCompleteMessage(Message):
    """Agent reporting job completion."""
    message_type: MessageType = MessageType.JOB_COMPLETE
    job_id: str
    success: bool

    class Config:
        use_enum_values = True


# Protocol Helpers
class ProtocolEncoder:
    """Encode messages for transmission."""

    @staticmethod
    def encode(message: Message) -> str:
        """Encode message to JSON string."""
        return message.json()

    @staticmethod
    def decode(data: str) -> Message:
        """Decode JSON string to message."""
        return Message.parse_raw(data)


class ProtocolValidator:
    """Validate protocol messages."""

    @staticmethod
    def validate_message(message: Message) -> bool:
        """Validate message format and requirements."""
        try:
            # Pydantic validation
            message.dict()

            # Additional validation rules
            if isinstance(message, Response) and not message.correlation_id:
                return False

            return True
        except Exception:
            return False

    @staticmethod
    def validate_sequence(messages: List[Message]) -> bool:
        """Validate a sequence of messages follows protocol."""
        # Check for request-response pairs
        requests = {}

        for msg in messages:
            if isinstance(msg, Request):
                requests[msg.message_id] = msg
            elif isinstance(msg, Response):
                if msg.correlation_id not in requests:
                    return False
                del requests[msg.correlation_id]

        # All requests should have responses
        return len(requests) == 0