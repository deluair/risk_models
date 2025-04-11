"""
Task List Manager for Risk Simulation Testing
Tracks and manages tasks for simulation testing process
"""
import os
import json
from pathlib import Path
from datetime import datetime
import argparse


class TaskList:
    """Manages a list of tasks for the risk simulation testing process"""
    
    def __init__(self, list_name="risk_model_testing"):
        """Initialize task list
        
        Args:
            list_name: Name of the task list
        """
        self.list_name = list_name
        self.file_path = Path(f"task_lists/{list_name}.json")
        self.file_path.parent.mkdir(exist_ok=True)
        
        # Load task list if exists, otherwise create new one
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                self.tasks = json.load(f)
        else:
            self.tasks = {
                "name": list_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "tasks": []
            }
            self._save()
    
    def add_task(self, description, category="general", priority="medium", dependencies=None):
        """Add a new task to the list
        
        Args:
            description: Description of the task
            category: Category of the task
            priority: Priority of the task (low, medium, high)
            dependencies: List of task IDs that must be completed before this task
        
        Returns:
            The ID of the new task
        """
        task_id = len(self.tasks["tasks"]) + 1
        
        if dependencies is None:
            dependencies = []
            
        new_task = {
            "id": task_id,
            "description": description,
            "category": category,
            "priority": priority,
            "dependencies": dependencies,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        self.tasks["tasks"].append(new_task)
        self.tasks["updated_at"] = datetime.now().isoformat()
        self._save()
        
        return task_id
    
    def update_task(self, task_id, **kwargs):
        """Update a task
        
        Args:
            task_id: ID of the task to update
            **kwargs: Fields to update
        
        Returns:
            True if the task was updated, False otherwise
        """
        for task in self.tasks["tasks"]:
            if task["id"] == task_id:
                for key, value in kwargs.items():
                    if key in task:
                        task[key] = value
                
                task["updated_at"] = datetime.now().isoformat()
                
                if "status" in kwargs and kwargs["status"] == "completed":
                    task["completed_at"] = datetime.now().isoformat()
                
                self.tasks["updated_at"] = datetime.now().isoformat()
                self._save()
                return True
        
        return False
    
    def mark_completed(self, task_id):
        """Mark a task as completed
        
        Args:
            task_id: ID of the task to mark as completed
        
        Returns:
            True if the task was updated, False otherwise
        """
        return self.update_task(task_id, status="completed")
    
    def get_tasks(self, status=None, category=None, priority=None):
        """Get tasks with optional filtering
        
        Args:
            status: Filter by status
            category: Filter by category
            priority: Filter by priority
        
        Returns:
            List of tasks matching the filters
        """
        filtered_tasks = self.tasks["tasks"]
        
        if status:
            filtered_tasks = [t for t in filtered_tasks if t["status"] == status]
        
        if category:
            filtered_tasks = [t for t in filtered_tasks if t["category"] == category]
        
        if priority:
            filtered_tasks = [t for t in filtered_tasks if t["priority"] == priority]
        
        return filtered_tasks
    
    def get_pending_tasks(self):
        """Get all pending tasks
        
        Returns:
            List of pending tasks
        """
        return self.get_tasks(status="pending")
    
    def get_completed_tasks(self):
        """Get all completed tasks
        
        Returns:
            List of completed tasks
        """
        return self.get_tasks(status="completed")
    
    def get_next_tasks(self):
        """Get tasks that can be done next (no pending dependencies)
        
        Returns:
            List of tasks that can be done next
        """
        pending_tasks = self.get_pending_tasks()
        completed_task_ids = [t["id"] for t in self.get_completed_tasks()]
        
        next_tasks = []
        for task in pending_tasks:
            if all(dep in completed_task_ids for dep in task["dependencies"]):
                next_tasks.append(task)
        
        return next_tasks
    
    def print_tasks(self, tasks=None, format_type="simple"):
        """Print tasks in a readable format
        
        Args:
            tasks: List of tasks to print (defaults to all tasks)
            format_type: Type of formatting to use
        """
        if tasks is None:
            tasks = self.tasks["tasks"]
        
        if not tasks:
            print("No tasks found.")
            return
        
        if format_type == "simple":
            for task in tasks:
                status_symbol = "✓" if task["status"] == "completed" else "□"
                print(f"[{status_symbol}] {task['id']}: {task['description']} ({task['priority']})")
        
        elif format_type == "detailed":
            for task in tasks:
                status_symbol = "✓" if task["status"] == "completed" else "□"
                priority_symbol = {
                    "low": "⬇️",
                    "medium": "⏺️",
                    "high": "⬆️"
                }.get(task["priority"], "")
                
                print(f"[{status_symbol}] {task['id']}: {priority_symbol} {task['description']}")
                print(f"    Category: {task['category']}")
                print(f"    Status: {task['status']}")
                
                if task["dependencies"]:
                    print(f"    Dependencies: {', '.join(map(str, task['dependencies']))}")
                
                if task["completed_at"]:
                    print(f"    Completed: {task['completed_at']}")
                print()
        
        elif format_type == "markdown":
            print("# Task List\n")
            
            for task in tasks:
                status_symbol = "✓" if task["status"] == "completed" else "□"
                priority_symbol = {
                    "low": "🔽",
                    "medium": "➖",
                    "high": "🔼"
                }.get(task["priority"], "")
                
                print(f"- [{status_symbol}] **{task['id']}**: {priority_symbol} {task['description']}")
                print(f"  - Category: `{task['category']}`")
                print(f"  - Status: `{task['status']}`")
                
                if task["dependencies"]:
                    deps = ", ".join(map(str, task["dependencies"]))
                    print(f"  - Dependencies: `{deps}`")
                
                if task["completed_at"]:
                    completed = task["completed_at"].split("T")[0]
                    print(f"  - Completed: `{completed}`")
                print()
    
    def generate_report(self, output_file=None):
        """Generate a report of the task list
        
        Args:
            output_file: Path to output file (defaults to task_lists/{list_name}_report.md)
        
        Returns:
            Path to the generated report
        """
        if output_file is None:
            output_file = Path(f"task_lists/{self.list_name}_report.md")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True)
        
        # Collect statistics
        total_tasks = len(self.tasks["tasks"])
        completed_tasks = len(self.get_completed_tasks())
        pending_tasks = len(self.get_pending_tasks())
        completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        categories = {}
        for task in self.tasks["tasks"]:
            category = task["category"]
            status = task["status"]
            
            if category not in categories:
                categories[category] = {"total": 0, "completed": 0}
            
            categories[category]["total"] += 1
            if status == "completed":
                categories[category]["completed"] += 1
        
        # Generate markdown report
        with open(output_file, "w") as f:
            f.write(f"# {self.list_name} Task Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Tasks: {total_tasks}\n")
            f.write(f"- Completed Tasks: {completed_tasks}\n")
            f.write(f"- Pending Tasks: {pending_tasks}\n")
            f.write(f"- Completion Rate: {completion_rate:.1f}%\n\n")
            
            f.write("## Categories\n\n")
            for category, stats in categories.items():
                cat_completion = (stats["completed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                f.write(f"- {category}: {stats['completed']}/{stats['total']} ({cat_completion:.1f}%)\n")
            
            f.write("\n## Pending Tasks\n\n")
            for task in self.get_pending_tasks():
                priority_symbol = {
                    "low": "🔽",
                    "medium": "➖",
                    "high": "🔼"
                }.get(task["priority"], "")
                
                f.write(f"- **{task['id']}**: {priority_symbol} {task['description']}\n")
                f.write(f"  - Category: `{task['category']}`\n")
                
                if task["dependencies"]:
                    deps = ", ".join(map(str, task["dependencies"]))
                    f.write(f"  - Dependencies: `{deps}`\n")
                
                f.write("\n")
            
            f.write("## Completed Tasks\n\n")
            for task in self.get_completed_tasks():
                f.write(f"- **{task['id']}**: {task['description']}\n")
                f.write(f"  - Category: `{task['category']}`\n")
                
                if task["completed_at"]:
                    completed = task["completed_at"].split("T")[0]
                    f.write(f"  - Completed: `{completed}`\n")
                
                f.write("\n")
        
        return output_file
    
    def _save(self):
        """Save the task list to disk"""
        with open(self.file_path, 'w') as f:
            json.dump(self.tasks, f, indent=2)


def create_default_task_list():
    """Create a default task list for risk model testing"""
    task_list = TaskList("risk_model_testing")
    
    # Data simulation tasks
    task_list.add_task(
        "Implement basic market data simulation",
        category="simulation",
        priority="high"
    )
    
    task_list.add_task(
        "Implement credit data simulation",
        category="simulation",
        priority="high"
    )
    
    task_list.add_task(
        "Implement market crash scenario",
        category="simulation",
        priority="medium",
        dependencies=[1]
    )
    
    task_list.add_task(
        "Implement credit deterioration scenario",
        category="simulation",
        priority="medium",
        dependencies=[2]
    )
    
    task_list.add_task(
        "Implement combined stress scenario",
        category="simulation",
        priority="medium",
        dependencies=[3, 4]
    )
    
    # Testing tasks
    task_list.add_task(
        "Implement simulation runner script",
        category="testing",
        priority="high",
        dependencies=[1, 2]
    )
    
    task_list.add_task(
        "Test market risk module with simulated data",
        category="testing",
        priority="high",
        dependencies=[6]
    )
    
    task_list.add_task(
        "Test credit risk module with simulated data",
        category="testing",
        priority="high",
        dependencies=[6]
    )
    
    task_list.add_task(
        "Test network risk module with simulated data",
        category="testing",
        priority="medium",
        dependencies=[6]
    )
    
    task_list.add_task(
        "Test systemic risk metrics with simulated data",
        category="testing",
        priority="medium",
        dependencies=[7, 8, 9]
    )
    
    # Validation tasks
    task_list.add_task(
        "Implement scenario comparison functionality",
        category="validation",
        priority="high",
        dependencies=[7, 8, 9]
    )
    
    task_list.add_task(
        "Compare results across different scenarios",
        category="validation",
        priority="high",
        dependencies=[3, 4, 5, 11]
    )
    
    task_list.add_task(
        "Validate market crash impact on risk metrics",
        category="validation",
        priority="medium",
        dependencies=[7, 12]
    )
    
    task_list.add_task(
        "Validate credit deterioration impact on risk metrics",
        category="validation",
        priority="medium", 
        dependencies=[8, 12]
    )
    
    task_list.add_task(
        "Document findings and insights from simulation",
        category="documentation",
        priority="medium",
        dependencies=[12, 13, 14]
    )
    
    return task_list


def main():
    parser = argparse.ArgumentParser(description="Task List Manager for Risk Simulation Testing")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize default task list")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=["pending", "completed", "all"], default="all", help="Filter by status")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--priority", choices=["low", "medium", "high"], help="Filter by priority")
    list_parser.add_argument("--format", choices=["simple", "detailed", "markdown"], default="simple", help="Output format")
    
    # Next command
    next_parser = subparsers.add_parser("next", help="Show next tasks")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("description", help="Task description")
    add_parser.add_argument("--category", default="general", help="Task category")
    add_parser.add_argument("--priority", choices=["low", "medium", "high"], default="medium", help="Task priority")
    add_parser.add_argument("--dependencies", type=int, nargs="+", help="Task dependencies (IDs)")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Mark a task as completed")
    complete_parser.add_argument("task_id", type=int, help="Task ID to mark as completed")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a task report")
    report_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    task_list = TaskList()
    
    if args.command == "init":
        create_default_task_list()
        print("Default task list created.")
    
    elif args.command == "list":
        status = None if args.status == "all" else args.status
        tasks = task_list.get_tasks(status=status, category=args.category, priority=args.priority)
        task_list.print_tasks(tasks, format_type=args.format)
    
    elif args.command == "next":
        next_tasks = task_list.get_next_tasks()
        print("Next tasks to complete:")
        task_list.print_tasks(next_tasks)
    
    elif args.command == "add":
        task_id = task_list.add_task(
            args.description,
            category=args.category,
            priority=args.priority,
            dependencies=args.dependencies
        )
        print(f"Task {task_id} added.")
    
    elif args.command == "complete":
        if task_list.mark_completed(args.task_id):
            print(f"Task {args.task_id} marked as completed.")
        else:
            print(f"Task {args.task_id} not found.")
    
    elif args.command == "report":
        report_path = task_list.generate_report(args.output)
        print(f"Report generated: {report_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 