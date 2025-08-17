"""
TestMaster Communication Module - Layer 2

File-based messaging and coordination inspired by framework analysis:
- Agency-Swarm: SharedState key-value store for status management
- Agency-Swarm: SendMessage validation for message integrity  
- Agency-Swarm: Thread-based conversation management

Provides:
- File-based messaging with Claude Code
- YAML/JSON structured message formats
- Bidirectional queue management
- Tag reading and synchronization
"""

from .claude_messenger import ClaudeMessenger, MessageType, MessagePriority
from .tag_reader import TagReader, FileTag, ModuleTag
from .message_queue import MessageQueue, QueueMessage

__all__ = [
    "ClaudeMessenger",
    "MessageType", 
    "MessagePriority",
    "TagReader",
    "FileTag",
    "ModuleTag", 
    "MessageQueue",
    "QueueMessage"
]