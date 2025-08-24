"""
Collaborative Testing Platform - Team collaboration and knowledge sharing

This platform provides:
- Team collaboration features for distributed testing
- Knowledge sharing and best practices repository  
- Code review integration for test quality
- Real-time collaboration and communication
- Team analytics and performance tracking
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
import uuid
import hashlib
from pathlib import Path
import concurrent.futures

# Mock Framework Imports for Testing
import pytest
from unittest.mock import Mock, patch, MagicMock
import unittest

class UserRole(Enum):
    ADMIN = "admin"
    LEAD = "lead"
    SENIOR = "senior"
    DEVELOPER = "developer"
    INTERN = "intern"
    OBSERVER = "observer"

class CollaborationType(Enum):
    PAIR_TESTING = "pair_testing"
    CODE_REVIEW = "code_review"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    MENTORING = "mentoring"
    TEAM_REVIEW = "team_review"

class ActivityType(Enum):
    TEST_CREATION = "test_creation"
    TEST_EXECUTION = "test_execution"
    CODE_REVIEW = "code_review"
    KNOWLEDGE_SHARE = "knowledge_share"
    ISSUE_RESOLUTION = "issue_resolution"
    BEST_PRACTICE = "best_practice"

class NotificationLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TeamMember:
    """Team member information and capabilities"""
    user_id: str
    username: str
    email: str
    role: UserRole
    skills: List[str]
    specializations: List[str]
    join_date: datetime
    last_active: Optional[datetime] = None
    performance_score: float = 0.0
    collaboration_score: float = 0.0
    contributions: int = 0
    preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CollaborationSession:
    """Active collaboration session"""
    session_id: str
    session_type: CollaborationType
    participants: List[str]
    repository: str
    focus_area: str
    start_time: datetime
    end_time: Optional[datetime] = None
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""
    outcomes: List[str] = field(default_factory=list)
    rating: Optional[float] = None

@dataclass
class KnowledgeArticle:
    """Knowledge base article"""
    article_id: str
    title: str
    content: str
    author: str
    category: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    views: int = 0
    likes: int = 0
    comments: List[Dict[str, Any]] = field(default_factory=list)
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced, expert

@dataclass
class TestReview:
    """Test code review session"""
    review_id: str
    test_file: str
    repository: str
    author: str
    reviewers: List[str]
    status: str  # pending, in_review, approved, rejected
    created_at: datetime
    comments: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: Optional[float] = None
    completion_time: Optional[datetime] = None

@dataclass
class TeamActivity:
    """Team activity tracking"""
    activity_id: str
    activity_type: ActivityType
    user_id: str
    repository: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 1.0

class NotificationManager:
    """Manages team notifications and communications"""
    
    def __init__(self):
        self.notifications: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.notification_history: List[Dict[str, Any]] = []
        
    def add_notification(self, user_id: str, title: str, message: str, 
                        level: NotificationLevel, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add notification for user"""
        notification_id = str(uuid.uuid4())
        notification = {
            "id": notification_id,
            "user_id": user_id,
            "title": title,
            "message": message,
            "level": level.value,
            "timestamp": datetime.now(),
            "read": False,
            "metadata": metadata or {}
        }
        
        self.notifications.append(notification)
        self.notification_history.append(notification.copy())
        
        # Trim old notifications
        if len(self.notifications) > 1000:
            self.notifications = self.notifications[-500:]
        
        return notification_id
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for user"""
        user_notifications = [
            notif for notif in self.notifications
            if notif["user_id"] == user_id
        ]
        
        if unread_only:
            user_notifications = [notif for notif in user_notifications if not notif["read"]]
        
        return sorted(user_notifications, key=lambda x: x["timestamp"], reverse=True)
    
    def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark notification as read"""
        for notification in self.notifications:
            if notification["id"] == notification_id and notification["user_id"] == user_id:
                notification["read"] = True
                return True
        return False
    
    def broadcast_notification(self, user_ids: List[str], title: str, message: str, 
                             level: NotificationLevel) -> List[str]:
        """Broadcast notification to multiple users"""
        notification_ids = []
        for user_id in user_ids:
            notif_id = self.add_notification(user_id, title, message, level)
            notification_ids.append(notif_id)
        return notification_ids
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Set notification preferences for user"""
        self.user_preferences[user_id] = preferences
    
    def should_notify_user(self, user_id: str, notification_type: str, level: NotificationLevel) -> bool:
        """Check if user should receive notification based on preferences"""
        prefs = self.user_preferences.get(user_id, {})
        
        # Default to notify unless explicitly disabled
        type_enabled = prefs.get(f"notify_{notification_type}", True)
        level_threshold = prefs.get("min_notification_level", NotificationLevel.INFO.value)
        
        level_values = {
            NotificationLevel.INFO.value: 1,
            NotificationLevel.WARNING.value: 2,
            NotificationLevel.ERROR.value: 3,
            NotificationLevel.CRITICAL.value: 4
        }
        
        return type_enabled and level_values.get(level.value, 1) >= level_values.get(level_threshold, 1)

class KnowledgeBase:
    """Collaborative knowledge base for testing best practices"""
    
    def __init__(self):
        self.articles: Dict[str, KnowledgeArticle] = {}
        self.categories: Dict[str, List[str]] = {}
        self.search_index: Dict[str, Set[str]] = {}  # word -> article_ids
        
    def create_article(self, title: str, content: str, author: str, category: str, 
                      tags: List[str], difficulty_level: str = "beginner") -> str:
        """Create new knowledge article"""
        article_id = str(uuid.uuid4())
        article = KnowledgeArticle(
            article_id=article_id,
            title=title,
            content=content,
            author=author,
            category=category,
            tags=tags,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            difficulty_level=difficulty_level
        )
        
        self.articles[article_id] = article
        
        # Update category index
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(article_id)
        
        # Update search index
        self._update_search_index(article)
        
        return article_id
    
    def update_article(self, article_id: str, updates: Dict[str, Any], user_id: str) -> bool:
        """Update existing article"""
        if article_id not in self.articles:
            return False
        
        article = self.articles[article_id]
        
        # Check permissions (author or admin can edit)
        # In real implementation, would check user role
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(article, key):
                setattr(article, key, value)
        
        article.updated_at = datetime.now()
        
        # Update search index
        self._update_search_index(article)
        
        return True
    
    def search_articles(self, query: str, category: Optional[str] = None, 
                       difficulty: Optional[str] = None, limit: int = 10) -> List[KnowledgeArticle]:
        """Search articles by query"""
        query_words = set(query.lower().split())
        matching_articles = set()
        
        # Find articles containing query words
        for word in query_words:
            if word in self.search_index:
                matching_articles.update(self.search_index[word])
        
        # Filter by category and difficulty
        results = []
        for article_id in matching_articles:
            if article_id not in self.articles:
                continue
            
            article = self.articles[article_id]
            
            if category and article.category != category:
                continue
            
            if difficulty and article.difficulty_level != difficulty:
                continue
            
            results.append(article)
        
        # Sort by relevance (simple scoring)
        results.sort(key=lambda a: self._calculate_relevance_score(a, query_words), reverse=True)
        
        return results[:limit]
    
    def get_popular_articles(self, category: Optional[str] = None, limit: int = 10) -> List[KnowledgeArticle]:
        """Get most popular articles"""
        articles = list(self.articles.values())
        
        if category:
            articles = [a for a in articles if a.category == category]
        
        # Sort by engagement score (views + likes)
        articles.sort(key=lambda a: a.views + (a.likes * 2), reverse=True)
        
        return articles[:limit]
    
    def add_comment(self, article_id: str, user_id: str, comment: str) -> bool:
        """Add comment to article"""
        if article_id not in self.articles:
            return False
        
        comment_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.now(),
            "likes": 0
        }
        
        self.articles[article_id].comments.append(comment_data)
        return True
    
    def like_article(self, article_id: str, user_id: str) -> bool:
        """Like an article"""
        if article_id not in self.articles:
            return False
        
        self.articles[article_id].likes += 1
        return True
    
    def view_article(self, article_id: str) -> Optional[KnowledgeArticle]:
        """View article (increments view count)"""
        if article_id not in self.articles:
            return None
        
        article = self.articles[article_id]
        article.views += 1
        return article
    
    def _update_search_index(self, article: KnowledgeArticle) -> None:
        """Update search index for article"""
        # Remove old index entries for this article
        for word_set in self.search_index.values():
            word_set.discard(article.article_id)
        
        # Index title and content words
        all_text = f"{article.title} {article.content} {' '.join(article.tags)}"
        words = set(all_text.lower().split())
        
        for word in words:
            if word not in self.search_index:
                self.search_index[word] = set()
            self.search_index[word].add(article.article_id)
    
    def _calculate_relevance_score(self, article: KnowledgeArticle, query_words: Set[str]) -> float:
        """Calculate relevance score for search results"""
        title_words = set(article.title.lower().split())
        content_words = set(article.content.lower().split())
        tag_words = set(word.lower() for word in article.tags)
        
        score = 0.0
        
        # Title matches are weighted heavily
        title_matches = len(query_words.intersection(title_words))
        score += title_matches * 3.0
        
        # Content matches
        content_matches = len(query_words.intersection(content_words))
        score += content_matches * 1.0
        
        # Tag matches
        tag_matches = len(query_words.intersection(tag_words))
        score += tag_matches * 2.0
        
        # Popularity boost
        score += (article.views * 0.01) + (article.likes * 0.05)
        
        return score

class CodeReviewSystem:
    """Code review system for test quality assurance"""
    
    def __init__(self):
        self.reviews: Dict[str, TestReview] = {}
        self.review_templates: Dict[str, Dict[str, Any]] = {}
        self.quality_metrics: Dict[str, List[float]] = {}
        
    def create_review(self, test_file: str, repository: str, author: str, 
                     reviewers: List[str]) -> str:
        """Create new test review"""
        review_id = str(uuid.uuid4())
        review = TestReview(
            review_id=review_id,
            test_file=test_file,
            repository=repository,
            author=author,
            reviewers=reviewers,
            status="pending",
            created_at=datetime.now()
        )
        
        self.reviews[review_id] = review
        return review_id
    
    def add_review_comment(self, review_id: str, reviewer: str, line_number: int, 
                          comment: str, suggestion: Optional[str] = None) -> bool:
        """Add comment to review"""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        
        comment_data = {
            "id": str(uuid.uuid4()),
            "reviewer": reviewer,
            "line_number": line_number,
            "comment": comment,
            "timestamp": datetime.now(),
            "resolved": False
        }
        
        review.comments.append(comment_data)
        
        if suggestion:
            suggestion_data = {
                "id": str(uuid.uuid4()),
                "reviewer": reviewer,
                "line_number": line_number,
                "suggestion": suggestion,
                "timestamp": datetime.now(),
                "applied": False
            }
            review.suggestions.append(suggestion_data)
        
        return True
    
    def approve_review(self, review_id: str, reviewer: str) -> bool:
        """Approve test review"""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        
        if reviewer not in review.reviewers:
            return False
        
        # Check if all reviewers have approved
        # For simplicity, assume single reviewer approval is sufficient
        review.status = "approved"
        review.completion_time = datetime.now()
        
        # Calculate quality score
        review.quality_score = self._calculate_quality_score(review)
        
        return True
    
    def reject_review(self, review_id: str, reviewer: str, reason: str) -> bool:
        """Reject test review"""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        
        if reviewer not in review.reviewers:
            return False
        
        review.status = "rejected"
        review.completion_time = datetime.now()
        
        # Add rejection reason as comment
        self.add_review_comment(review_id, reviewer, 0, f"Review rejected: {reason}")
        
        return True
    
    def get_review_statistics(self, repository: Optional[str] = None, 
                            time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get review statistics"""
        reviews = list(self.reviews.values())
        
        if repository:
            reviews = [r for r in reviews if r.repository == repository]
        
        if time_period:
            cutoff = datetime.now() - time_period
            reviews = [r for r in reviews if r.created_at > cutoff]
        
        if not reviews:
            return {"error": "No reviews found for criteria"}
        
        total_reviews = len(reviews)
        approved_reviews = len([r for r in reviews if r.status == "approved"])
        rejected_reviews = len([r for r in reviews if r.status == "rejected"])
        
        # Calculate average review time
        completed_reviews = [r for r in reviews if r.completion_time]
        avg_review_time = None
        if completed_reviews:
            total_time = sum(
                (r.completion_time - r.created_at).total_seconds()
                for r in completed_reviews
            )
            avg_review_time = total_time / len(completed_reviews) / 3600  # hours
        
        # Quality score distribution
        quality_scores = [r.quality_score for r in reviews if r.quality_score is not None]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else None
        
        return {
            "total_reviews": total_reviews,
            "approved_reviews": approved_reviews,
            "rejected_reviews": rejected_reviews,
            "pending_reviews": total_reviews - approved_reviews - rejected_reviews,
            "approval_rate": approved_reviews / max(1, total_reviews),
            "average_review_time_hours": avg_review_time,
            "average_quality_score": avg_quality_score,
            "reviews_by_author": self._count_by_field(reviews, "author"),
            "reviews_by_repository": self._count_by_field(reviews, "repository")
        }
    
    def _calculate_quality_score(self, review: TestReview) -> float:
        """Calculate quality score based on review content"""
        base_score = 7.0  # Start with 7/10
        
        # Deduct points for issues found
        critical_issues = len([c for c in review.comments if "critical" in c.get("comment", "").lower()])
        major_issues = len([c for c in review.comments if "major" in c.get("comment", "").lower()])
        minor_issues = len([c for c in review.comments if "minor" in c.get("comment", "").lower()])
        
        base_score -= critical_issues * 2.0
        base_score -= major_issues * 1.0
        base_score -= minor_issues * 0.3
        
        # Add points for good practices mentioned
        good_practices = len([c for c in review.comments if any(
            phrase in c.get("comment", "").lower() 
            for phrase in ["good", "excellent", "well done", "nice"]
        )])
        base_score += good_practices * 0.2
        
        return max(0.0, min(10.0, base_score))
    
    def _count_by_field(self, reviews: List[TestReview], field: str) -> Dict[str, int]:
        """Count reviews by field value"""
        counts = {}
        for review in reviews:
            value = getattr(review, field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

class TeamAnalytics:
    """Team performance analytics and insights"""
    
    def __init__(self):
        self.activities: List[TeamActivity] = []
        self.collaboration_sessions: List[CollaborationSession] = []
        self.team_members: Dict[str, TeamMember] = {}
        
    def record_activity(self, activity: TeamActivity) -> None:
        """Record team member activity"""
        self.activities.append(activity)
        
        # Trim old activities (keep last 10000)
        if len(self.activities) > 10000:
            self.activities = self.activities[-5000:]
        
        # Update member performance
        self._update_member_performance(activity)
    
    def record_collaboration_session(self, session: CollaborationSession) -> None:
        """Record collaboration session"""
        self.collaboration_sessions.append(session)
        
        # Update collaboration scores for participants
        for participant in session.participants:
            if participant in self.team_members:
                member = self.team_members[participant]
                member.collaboration_score += 1.0
                
                if session.rating:
                    member.collaboration_score += session.rating * 0.5
    
    def add_team_member(self, member: TeamMember) -> None:
        """Add team member"""
        self.team_members[member.user_id] = member
    
    def get_team_performance(self, time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get team performance metrics"""
        activities = self.activities
        sessions = self.collaboration_sessions
        
        if time_period:
            cutoff = datetime.now() - time_period
            activities = [a for a in activities if a.timestamp > cutoff]
            sessions = [s for s in sessions if s.start_time > cutoff]
        
        # Activity analysis
        activity_counts = {}
        for activity in activities:
            activity_type = activity.activity_type.value
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        # Collaboration analysis
        collaboration_types = {}
        for session in sessions:
            session_type = session.session_type.value
            collaboration_types[session_type] = collaboration_types.get(session_type, 0) + 1
        
        # Member performance rankings
        member_scores = []
        for member in self.team_members.values():
            total_score = member.performance_score + member.collaboration_score
            member_scores.append({
                "user_id": member.user_id,
                "username": member.username,
                "role": member.role.value,
                "performance_score": member.performance_score,
                "collaboration_score": member.collaboration_score,
                "total_score": total_score,
                "contributions": member.contributions
            })
        
        member_scores.sort(key=lambda x: x["total_score"], reverse=True)
        
        return {
            "time_period_days": time_period.days if time_period else "all_time",
            "total_activities": len(activities),
            "total_collaborations": len(sessions),
            "activity_breakdown": activity_counts,
            "collaboration_breakdown": collaboration_types,
            "top_performers": member_scores[:10],
            "team_size": len(self.team_members),
            "average_performance_score": sum(m.performance_score for m in self.team_members.values()) / max(1, len(self.team_members)),
            "average_collaboration_score": sum(m.collaboration_score for m in self.team_members.values()) / max(1, len(self.team_members))
        }
    
    def get_individual_insights(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get insights for individual team member"""
        if user_id not in self.team_members:
            return None
        
        member = self.team_members[user_id]
        
        # Get user activities from last 30 days
        recent_cutoff = datetime.now() - timedelta(days=30)
        user_activities = [a for a in self.activities if a.user_id == user_id and a.timestamp > recent_cutoff]
        
        # Get collaboration sessions
        user_sessions = [s for s in self.collaboration_sessions if user_id in s.participants]
        
        # Calculate trends
        activity_trend = self._calculate_activity_trend(user_activities)
        
        return {
            "user_id": user_id,
            "username": member.username,
            "role": member.role.value,
            "skills": member.skills,
            "specializations": member.specializations,
            "performance_score": member.performance_score,
            "collaboration_score": member.collaboration_score,
            "total_contributions": member.contributions,
            "recent_activities": len(user_activities),
            "recent_collaborations": len([s for s in user_sessions if s.start_time > recent_cutoff]),
            "activity_trend": activity_trend,
            "strengths": self._identify_strengths(member, user_activities),
            "recommendations": self._generate_recommendations(member, user_activities)
        }
    
    def _update_member_performance(self, activity: TeamActivity) -> None:
        """Update member performance based on activity"""
        if activity.user_id not in self.team_members:
            return
        
        member = self.team_members[activity.user_id]
        member.performance_score += activity.impact_score
        member.contributions += 1
        member.last_active = activity.timestamp
    
    def _calculate_activity_trend(self, activities: List[TeamActivity]) -> str:
        """Calculate activity trend"""
        if len(activities) < 7:
            return "insufficient_data"
        
        # Compare last week to previous week
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)
        
        last_week = len([a for a in activities if week_ago <= a.timestamp <= now])
        prev_week = len([a for a in activities if two_weeks_ago <= a.timestamp < week_ago])
        
        if prev_week == 0:
            return "new_activity"
        
        change_ratio = last_week / prev_week
        
        if change_ratio > 1.2:
            return "increasing"
        elif change_ratio < 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_strengths(self, member: TeamMember, activities: List[TeamActivity]) -> List[str]:
        """Identify member strengths based on activities"""
        strengths = []
        
        # Analyze activity types
        activity_counts = {}
        for activity in activities:
            activity_type = activity.activity_type.value
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
        
        # Find top activity types
        sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
        
        for activity_type, count in sorted_activities[:3]:
            if count > 5:  # Significant activity
                strengths.append(activity_type.replace("_", " ").title())
        
        # Add role-based strengths
        if member.role in [UserRole.LEAD, UserRole.SENIOR]:
            strengths.append("Leadership")
        
        if member.collaboration_score > 10:
            strengths.append("Collaboration")
        
        return strengths
    
    def _generate_recommendations(self, member: TeamMember, activities: List[TeamActivity]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Activity-based recommendations
        if len(activities) < 10:
            recommendations.append("Increase testing activity participation")
        
        if member.collaboration_score < 5:
            recommendations.append("Participate more in team collaboration sessions")
        
        # Role-based recommendations
        if member.role == UserRole.INTERN:
            recommendations.append("Consider pairing with senior members for mentoring")
        
        if member.role in [UserRole.SENIOR, UserRole.LEAD] and member.collaboration_score < 15:
            recommendations.append("Lead more knowledge sharing sessions")
        
        return recommendations

class CollaborativeTestingPlatform:
    """Main platform for collaborative testing features"""
    
    def __init__(self):
        self.notification_manager = NotificationManager()
        self.knowledge_base = KnowledgeBase()
        self.code_review_system = CodeReviewSystem()
        self.team_analytics = TeamAnalytics()
        
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.platform_settings: Dict[str, Any] = self._default_settings()
        
    def _default_settings(self) -> Dict[str, Any]:
        """Default platform settings"""
        return {
            "enable_real_time_collaboration": True,
            "enable_automatic_reviews": True,
            "require_code_review": True,
            "knowledge_base_moderation": False,
            "team_analytics_retention_days": 90
        }
    
    def register_team_member(self, member: TeamMember) -> bool:
        """Register new team member"""
        try:
            self.team_analytics.add_team_member(member)
            
            # Send welcome notification
            self.notification_manager.add_notification(
                member.user_id,
                "Welcome to TestMaster",
                f"Welcome {member.username}! You've been added to the collaborative testing platform.",
                NotificationLevel.INFO
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to register team member {member.user_id}: {e}")
            return False
    
    def start_collaboration_session(self, session_type: CollaborationType, 
                                  participants: List[str], repository: str, 
                                  focus_area: str) -> Optional[str]:
        """Start new collaboration session"""
        session_id = str(uuid.uuid4())
        session = CollaborationSession(
            session_id=session_id,
            session_type=session_type,
            participants=participants,
            repository=repository,
            focus_area=focus_area,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        # Notify participants
        for participant in participants:
            self.notification_manager.add_notification(
                participant,
                f"Collaboration Session Started",
                f"You've been invited to a {session_type.value} session for {repository}",
                NotificationLevel.INFO,
                {"session_id": session_id}
            )
        
        return session_id
    
    def end_collaboration_session(self, session_id: str, outcomes: List[str], 
                                 rating: Optional[float] = None, notes: str = "") -> bool:
        """End collaboration session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        session.outcomes = outcomes
        session.rating = rating
        session.notes = notes
        
        # Record session in analytics
        self.team_analytics.record_collaboration_session(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Notify participants of completion
        for participant in session.participants:
            self.notification_manager.add_notification(
                participant,
                "Collaboration Session Completed",
                f"The {session.session_type.value} session has been completed",
                NotificationLevel.INFO
            )
        
        return True
    
    def submit_test_for_review(self, test_file: str, repository: str, 
                             author: str, reviewers: List[str]) -> Optional[str]:
        """Submit test for code review"""
        if not self.platform_settings.get("require_code_review", True):
            return None
        
        review_id = self.code_review_system.create_review(
            test_file, repository, author, reviewers
        )
        
        # Notify reviewers
        for reviewer in reviewers:
            self.notification_manager.add_notification(
                reviewer,
                "Code Review Request",
                f"Please review test file {test_file} in {repository}",
                NotificationLevel.INFO,
                {"review_id": review_id}
            )
        
        return review_id
    
    def share_knowledge(self, title: str, content: str, author: str, 
                       category: str, tags: List[str]) -> str:
        """Share knowledge article"""
        article_id = self.knowledge_base.create_article(
            title, content, author, category, tags
        )
        
        # Notify team about new knowledge sharing
        team_members = list(self.team_analytics.team_members.keys())
        self.notification_manager.broadcast_notification(
            team_members,
            "New Knowledge Article",
            f"New article '{title}' has been shared in {category}",
            NotificationLevel.INFO
        )
        
        # Record activity
        activity = TeamActivity(
            activity_id=str(uuid.uuid4()),
            activity_type=ActivityType.KNOWLEDGE_SHARE,
            user_id=author,
            repository="knowledge_base",
            description=f"Shared article: {title}",
            timestamp=datetime.now(),
            impact_score=2.0  # Higher impact for knowledge sharing
        )
        self.team_analytics.record_activity(activity)
        
        return article_id
    
    def get_platform_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get personalized platform dashboard"""
        # Get user notifications
        notifications = self.notification_manager.get_user_notifications(user_id, unread_only=True)
        
        # Get user insights
        user_insights = self.team_analytics.get_individual_insights(user_id)
        
        # Get popular knowledge articles
        popular_articles = self.knowledge_base.get_popular_articles(limit=5)
        
        # Get pending reviews for user
        pending_reviews = [
            review for review in self.code_review_system.reviews.values()
            if user_id in review.reviewers and review.status == "pending"
        ]
        
        # Get active collaboration sessions
        user_active_sessions = [
            session for session in self.active_sessions.values()
            if user_id in session.participants
        ]
        
        return {
            "user_id": user_id,
            "unread_notifications": len(notifications),
            "notifications_preview": notifications[:5],
            "user_insights": user_insights,
            "popular_knowledge": [
                {
                    "title": article.title,
                    "category": article.category,
                    "views": article.views,
                    "likes": article.likes
                }
                for article in popular_articles
            ],
            "pending_reviews": len(pending_reviews),
            "active_collaborations": len(user_active_sessions),
            "platform_stats": {
                "total_team_members": len(self.team_analytics.team_members),
                "total_knowledge_articles": len(self.knowledge_base.articles),
                "total_reviews": len(self.code_review_system.reviews),
                "active_sessions": len(self.active_sessions)
            }
        }
    
    def get_team_overview(self) -> Dict[str, Any]:
        """Get team overview and statistics"""
        # Team performance
        team_performance = self.team_analytics.get_team_performance(timedelta(days=30))
        
        # Review statistics
        review_stats = self.code_review_system.get_review_statistics(time_period=timedelta(days=30))
        
        # Knowledge base stats
        kb_stats = {
            "total_articles": len(self.knowledge_base.articles),
            "categories": len(self.knowledge_base.categories),
            "total_views": sum(article.views for article in self.knowledge_base.articles.values()),
            "total_likes": sum(article.likes for article in self.knowledge_base.articles.values())
        }
        
        return {
            "team_performance": team_performance,
            "code_review_stats": review_stats,
            "knowledge_base_stats": kb_stats,
            "active_collaborations": len(self.active_sessions),
            "platform_health": "good"  # Could be calculated based on various factors
        }


# Comprehensive Test Suite
class TestCollaborativeTestingPlatform(unittest.TestCase):
    
    def setUp(self):
        self.platform = CollaborativeTestingPlatform()
        
        # Add test team member
        self.test_member = TeamMember(
            user_id="test_user_001",
            username="test_developer",
            email="test@example.com",
            role=UserRole.DEVELOPER,
            skills=["python", "testing", "pytest"],
            specializations=["unit_testing", "integration_testing"],
            join_date=datetime.now()
        )
        self.platform.register_team_member(self.test_member)
        
    def test_platform_initialization(self):
        """Test platform initialization"""
        self.assertIsNotNone(self.platform.notification_manager)
        self.assertIsNotNone(self.platform.knowledge_base)
        self.assertIsNotNone(self.platform.code_review_system)
        self.assertIsNotNone(self.platform.team_analytics)
        
    def test_team_member_registration(self):
        """Test team member registration"""
        new_member = TeamMember(
            user_id="test_user_002",
            username="senior_dev",
            email="senior@example.com",
            role=UserRole.SENIOR,
            skills=["python", "javascript", "testing"],
            specializations=["test_automation"],
            join_date=datetime.now()
        )
        
        success = self.platform.register_team_member(new_member)
        self.assertTrue(success)
        self.assertIn("test_user_002", self.platform.team_analytics.team_members)
        
    def test_collaboration_session(self):
        """Test collaboration session management"""
        participants = ["test_user_001", "test_user_002"]
        session_id = self.platform.start_collaboration_session(
            CollaborationType.PAIR_TESTING,
            participants,
            "test_repository",
            "unit_testing"
        )
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.platform.active_sessions)
        
        # End session
        success = self.platform.end_collaboration_session(
            session_id,
            ["Implemented 5 new test cases", "Fixed 2 failing tests"],
            8.5,
            "Productive session"
        )
        self.assertTrue(success)
        self.assertNotIn(session_id, self.platform.active_sessions)
        
    def test_knowledge_sharing(self):
        """Test knowledge sharing functionality"""
        article_id = self.platform.share_knowledge(
            "Best Practices for Pytest Fixtures",
            "Here are some best practices for using pytest fixtures...",
            "test_user_001",
            "Testing Best Practices",
            ["pytest", "fixtures", "best-practices"]
        )
        
        self.assertIsNotNone(article_id)
        self.assertIn(article_id, self.platform.knowledge_base.articles)
        
        # Test article search
        results = self.platform.knowledge_base.search_articles("pytest fixtures")
        self.assertGreater(len(results), 0)
        
    def test_code_review_system(self):
        """Test code review functionality"""
        review_id = self.platform.submit_test_for_review(
            "test_user_service.py",
            "user_service_repo",
            "test_user_001",
            ["test_user_002"]
        )
        
        self.assertIsNotNone(review_id)
        self.assertIn(review_id, self.platform.code_review_system.reviews)
        
        # Add review comment
        success = self.platform.code_review_system.add_review_comment(
            review_id,
            "test_user_002",
            25,
            "Consider adding more edge case tests here",
            "Add test for empty input validation"
        )
        self.assertTrue(success)
        
        # Approve review
        success = self.platform.code_review_system.approve_review(review_id, "test_user_002")
        self.assertTrue(success)
        
    def test_notification_system(self):
        """Test notification system"""
        notif_id = self.platform.notification_manager.add_notification(
            "test_user_001",
            "Test Notification",
            "This is a test notification",
            NotificationLevel.INFO
        )
        
        self.assertIsNotNone(notif_id)
        
        # Get user notifications
        notifications = self.platform.notification_manager.get_user_notifications("test_user_001")
        self.assertGreater(len(notifications), 0)
        
        # Mark as read
        success = self.platform.notification_manager.mark_notification_read(notif_id, "test_user_001")
        self.assertTrue(success)
        
    def test_team_analytics(self):
        """Test team analytics functionality"""
        # Record some activities
        activity = TeamActivity(
            activity_id="activity_001",
            activity_type=ActivityType.TEST_CREATION,
            user_id="test_user_001",
            repository="test_repo",
            description="Created unit tests for user service",
            timestamp=datetime.now(),
            impact_score=3.0
        )
        
        self.platform.team_analytics.record_activity(activity)
        
        # Get team performance
        performance = self.platform.team_analytics.get_team_performance()
        self.assertIn("total_activities", performance)
        self.assertEqual(performance["total_activities"], 1)
        
        # Get individual insights
        insights = self.platform.team_analytics.get_individual_insights("test_user_001")
        self.assertIsNotNone(insights)
        self.assertEqual(insights["user_id"], "test_user_001")
        
    def test_platform_dashboard(self):
        """Test platform dashboard generation"""
        dashboard = self.platform.get_platform_dashboard("test_user_001")
        
        self.assertIn("user_id", dashboard)
        self.assertIn("unread_notifications", dashboard)
        self.assertIn("user_insights", dashboard)
        self.assertIn("platform_stats", dashboard)
        
    def test_team_overview(self):
        """Test team overview generation"""
        overview = self.platform.get_team_overview()
        
        self.assertIn("team_performance", overview)
        self.assertIn("code_review_stats", overview)
        self.assertIn("knowledge_base_stats", overview)
        self.assertIn("platform_health", overview)


if __name__ == "__main__":
    # Demo usage
    platform = CollaborativeTestingPlatform()
    
    # Register team members
    team_members = [
        TeamMember(
            user_id="lead_001",
            username="team_lead",
            email="lead@testmaster.com",
            role=UserRole.LEAD,
            skills=["python", "javascript", "testing", "management"],
            specializations=["test_strategy", "team_leadership"],
            join_date=datetime.now()
        ),
        TeamMember(
            user_id="senior_001",
            username="senior_dev",
            email="senior@testmaster.com",
            role=UserRole.SENIOR,
            skills=["python", "pytest", "automation"],
            specializations=["test_automation", "ci_cd"],
            join_date=datetime.now()
        ),
        TeamMember(
            user_id="dev_001",
            username="developer",
            email="dev@testmaster.com",
            role=UserRole.DEVELOPER,
            skills=["python", "testing"],
            specializations=["unit_testing"],
            join_date=datetime.now()
        )
    ]
    
    for member in team_members:
        platform.register_team_member(member)
    
    # Start collaboration session
    session_id = platform.start_collaboration_session(
        CollaborationType.CODE_REVIEW,
        ["lead_001", "senior_001", "dev_001"],
        "testmaster_core",
        "review_new_testing_framework"
    )
    print(f"Started collaboration session: {session_id}")
    
    # Share knowledge
    article_id = platform.share_knowledge(
        "Advanced Pytest Techniques",
        "This article covers advanced pytest techniques including fixtures, parametrization, and custom plugins...",
        "senior_001",
        "Testing Advanced Topics",
        ["pytest", "advanced", "fixtures", "parametrization"]
    )
    print(f"Shared knowledge article: {article_id}")
    
    # Submit code for review
    review_id = platform.submit_test_for_review(
        "test_collaborative_platform.py",
        "testmaster_core",
        "dev_001",
        ["senior_001", "lead_001"]
    )
    print(f"Submitted code for review: {review_id}")
    
    # Get team overview
    overview = platform.get_team_overview()
    print(f"Team Overview: {json.dumps(overview, indent=2, default=str)}")
    
    print("Collaborative Testing Platform Demo Complete")
    
    # Run tests
    pytest.main([__file__, "-v"])