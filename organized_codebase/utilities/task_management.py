from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any


class TaskStatus(Enum):
    """Enumeration for task status"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Enumeration for task priority"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class Task:
    """Class representing a single task"""
    
    def __init__(
        self, 
        title: str, 
        description: str = "", 
        priority: Priority = Priority.MEDIUM,
        due_date: Optional[datetime] = None,
        task_id: Optional[str] = None
    ):
        """
        Initialize a new task
        
        Args:
            title: The title of the task
            description: Detailed description of the task
            priority: Priority level of the task
            due_date: Optional due date for the task
            task_id: Optional unique identifier for the task
        """
        self.title = title
        self.description = description
        self.priority = priority
        self.due_date = due_date
        self.status = TaskStatus.TODO
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.completed_at = None
        
        # Generate a unique ID if not provided
        self.id = task_id or f"task_{int(datetime.now().timestamp())}"
    
    def mark_complete(self):
        """Mark the task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def mark_in_progress(self):
        """Mark the task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = datetime.now()
    
    def mark_todo(self):
        """Mark the task as to do"""
        self.status = TaskStatus.TODO
        self.updated_at = datetime.now()
    
    def cancel(self):
        """Cancel the task"""
        self.status = TaskStatus.CANCELLED
        self.updated_at = datetime.now()
    
    def update(self, title: Optional[str] = None, description: Optional[str] = None, 
               priority: Optional[Priority] = None, due_date: Optional[datetime] = None):
        """Update task details"""
        if title is not None:
            self.title = title
        if description is not None:
            self.description = description
        if priority is not None:
            self.priority = priority
        if due_date is not None:
            self.due_date = due_date
        
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.value,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def __str__(self) -> str:
        """String representation of the task"""
        status_emoji = {
            TaskStatus.TODO: "📝",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.CANCELLED: "❌"
        }
        
        priority_emoji = {
            Priority.LOW: "🟢",
            Priority.MEDIUM: "🟡",
            Priority.HIGH: "🟠",
            Priority.URGENT: "🔴"
        }
        
        due_str = f" (Due: {self.due_date.strftime('%Y-%m-%d')})" if self.due_date else ""
        return f"{status_emoji[self.status]} {priority_emoji[self.priority]} {self.title}{due_str}"


class TaskList:
    """Class representing a collection of tasks"""
    
    def __init__(self, name: str):
        """
        Initialize a new task list
        
        Args:
            name: Name of the task list
        """
        self.name = name
        self.tasks: List[Task] = []
        self.created_at = datetime.now()
    
    def add_task(self, task: Task) -> str:
        """
        Add a task to the task list
        
        Args:
            task: The task to add
            
        Returns:
            The ID of the added task
        """
        self.tasks.append(task)
        return task.id
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the task list
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            True if the task was found and removed, False otherwise
        """
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                self.tasks.pop(i)
                return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            The task if found, None otherwise
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def filter_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Filter tasks by status
        
        Args:
            status: The status to filter by
            
        Returns:
            List of tasks with the specified status
        """
        return [task for task in self.tasks if task.status == status]
    
    def filter_by_priority(self, priority: Priority) -> List[Task]:
        """
        Filter tasks by priority
        
        Args:
            priority: The priority to filter by
            
        Returns:
            List of tasks with the specified priority
        """
        return [task for task in self.tasks if task.priority == priority]
    
    def get_overdue_tasks(self) -> List[Task]:
        """
        Get tasks that are past their due date and not completed
        
        Returns:
            List of overdue tasks
        """
        now = datetime.now()
        return [
            task for task in self.tasks 
            if task.due_date and task.due_date < now and task.status != TaskStatus.COMPLETED
        ]
    
    def get_upcoming_tasks(self, days: int = 7) -> List[Task]:
        """
        Get tasks that are due within the specified number of days
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of upcoming tasks
        """
        now = datetime.now()
        return [
            task for task in self.tasks 
            if task.due_date and now <= task.due_date <= now.replace(day=now.day + days)
        ]
    
    def sort_by_priority(self) -> List[Task]:
        """
        Sort tasks by priority (highest first)
        
        Returns:
            List of tasks sorted by priority
        """
        return sorted(self.tasks, key=lambda task: task.priority.value, reverse=True)
    
    def sort_by_due_date(self) -> List[Task]:
        """
        Sort tasks by due date (earliest first)
        
        Returns:
            List of tasks sorted by due date
        """
        # Tasks without due dates go to the end
        return sorted(
            self.tasks, 
            key=lambda task: (task.due_date is None, task.due_date)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the task list
        
        Returns:
            Dictionary containing task statistics
        """
        total = len(self.tasks)
        if total == 0:
            return {
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "todo": 0,
                "cancelled": 0,
                "completion_rate": 0
            }
        
        completed = len(self.filter_by_status(TaskStatus.COMPLETED))
        in_progress = len(self.filter_by_status(TaskStatus.IN_PROGRESS))
        todo = len(self.filter_by_status(TaskStatus.TODO))
        cancelled = len(self.filter_by_status(TaskStatus.CANCELLED))
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "todo": todo,
            "cancelled": cancelled,
            "completion_rate": round(completed / (total - cancelled) * 100, 1)
        }
    
    def __str__(self) -> str:
        """String representation of the task list"""
        stats = self.get_stats()
        return f"TaskList '{self.name}': {stats['total']} tasks, {stats['completed']} completed ({stats['completion_rate']}%)"


class User:
    """Class representing a user who owns task lists"""
    
    def __init__(self, username: str, email: str, full_name: str = ""):
        """
        Initialize a new user
        
        Args:
            username: Unique username
            email: User's email address
            full_name: User's full name
        """
        self.username = username
        self.email = email
        self.full_name = full_name
        self.task_lists: Dict[str, TaskList] = {}
        self.created_at = datetime.now()
    
    def create_task_list(self, name: str) -> str:
        """
        Create a new task list for the user
        
        Args:
            name: Name of the task list
            
        Returns:
            Name of the created task list
        """
        if name in self.task_lists:
            raise ValueError(f"Task list '{name}' already exists")
        
        self.task_lists[name] = TaskList(name)
        return name
    
    def delete_task_list(self, name: str) -> bool:
        """
        Delete a task list
        
        Args:
            name: Name of the task list to delete
            
        Returns:
            True if the task list was found and deleted, False otherwise
        """
        if name in self.task_lists:
            del self.task_lists[name]
            return True
        return False
    
    def get_task_list(self, name: str) -> Optional[TaskList]:
        """
        Get a task list by name
        
        Args:
            name: Name of the task list
            
        Returns:
            The task list if found, None otherwise
        """
        return self.task_lists.get(name)
    
    def add_task_to_list(self, list_name: str, task: Task) -> Optional[str]:
        """
        Add a task to a specific task list
        
        Args:
            list_name: Name of the task list
            task: The task to add
            
        Returns:
            The ID of the added task if successful, None otherwise
        """
        task_list = self.get_task_list(list_name)
        if task_list:
            return task_list.add_task(task)
        return None
    
    def get_all_tasks(self) -> List[Task]:
        """
        Get all tasks across all task lists
        
        Returns:
            List of all tasks
        """
        all_tasks = []
        for task_list in self.task_lists.values():
            all_tasks.extend(task_list.tasks)
        return all_tasks
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Get all tasks with a specific status across all task lists
        
        Args:
            status: The status to filter by
            
        Returns:
            List of tasks with the specified status
        """
        tasks = []
        for task_list in self.task_lists.values():
            tasks.extend(task_list.filter_by_status(status))
        return tasks
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics across all task lists
        
        Returns:
            Dictionary containing overall task statistics
        """
        all_tasks = self.get_all_tasks()
        total = len(all_tasks)
        
        if total == 0:
            return {
                "total_task_lists": len(self.task_lists),
                "total_tasks": 0,
                "completed": 0,
                "in_progress": 0,
                "todo": 0,
                "cancelled": 0,
                "completion_rate": 0
            }
        
        completed = len([task for task in all_tasks if task.status == TaskStatus.COMPLETED])
        in_progress = len([task for task in all_tasks if task.status == TaskStatus.IN_PROGRESS])
        todo = len([task for task in all_tasks if task.status == TaskStatus.TODO])
        cancelled = len([task for task in all_tasks if task.status == TaskStatus.CANCELLED])
        
        return {
            "total_task_lists": len(self.task_lists),
            "total_tasks": total,
            "completed": completed,
            "in_progress": in_progress,
            "todo": todo,
            "cancelled": cancelled,
            "completion_rate": round(completed / (total - cancelled) * 100, 1)
        }
    
    def __str__(self) -> str:
        """String representation of the user"""
        stats = self.get_overall_stats()
        return f"User '{self.username}': {stats['total_task_lists']} lists, {stats['total_tasks']} tasks"
