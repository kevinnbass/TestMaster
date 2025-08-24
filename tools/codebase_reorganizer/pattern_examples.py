#!/usr/bin/env python3
"""
Pattern Detection Examples
==========================

Example classes demonstrating common design patterns for testing pattern detection.
"""

import asyncio
from typing import Optional


class User:
    """Example User class for pattern detection testing"""
    id: int
    name: str
    email: str

    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email


class UserRepository:
    """Example Repository pattern implementation"""

    def __init__(self, db_connection):
        self.db = db_connection

    def save(self, user: User) -> bool:
        """Save user to database (Repository pattern example)"""
        # Save user to database
        return True

    def find_by_id(self, user_id: int) -> User:
        """Find user by ID (Repository pattern example)"""
        # Find user by ID
        return User(id=user_id, name="John", email="john@example.com")


class UserService:
    """Example Service pattern implementation"""

    def __init__(self, repository: UserRepository):
        self.repository = repository

    def create_user(self, name: str, email: str) -> User:
        """Create a new user (Service pattern example)"""
        user = User(id=123, name=name, email=email)
        self.repository.save(user)
        return user

    async def get_user_async(self, user_id: int) -> User:
        """Get user asynchronously (Async pattern example)"""
        # Simulate async operation
        await asyncio.sleep(0.1)
        return self.repository.find_by_id(user_id)


# Singleton pattern for configuration
class Config:
    """Example Singleton pattern implementation"""
    _instance = None

    def __init__(self):
        if Config._instance is not None:
            raise Exception("Config is a singleton class")
        self.settings = {}

    @classmethod
    def get_instance(cls):
        """Get singleton instance (Singleton pattern example)"""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance


def demonstrate_patterns():
    """
    Demonstrate various design patterns using the example classes.

    This function serves as a test case for pattern detection and shows
    how different design patterns work together.
    """
    # Get singleton config instance
    config = Config.get_instance()

    # Create repository and service (Dependency Injection pattern)
    repo = UserRepository(None)
    service = UserService(repo)

    # Create a user (Service pattern)
    user = service.create_user("Alice", "alice@example.com")
    print(f"Created user: {user.name} ({user.email})")

    return user


if __name__ == "__main__":
    demonstrate_patterns()

