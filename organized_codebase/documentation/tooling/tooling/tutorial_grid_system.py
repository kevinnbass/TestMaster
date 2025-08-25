"""
Tutorial Grid System

Creates visual card-based tutorial navigation with scenario-based examples
and graduated complexity based on AutoGen's grid-based tutorial approach.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TutorialDifficulty(Enum):
    """Tutorial difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"  
    ADVANCED = "advanced"
    EXPERT = "expert"


class TutorialCategory(Enum):
    """Tutorial categories."""
    GETTING_STARTED = "getting_started"
    BASIC_CONCEPTS = "basic_concepts"
    REAL_WORLD = "real_world"
    ADVANCED_TOPICS = "advanced_topics"
    INTEGRATIONS = "integrations"
    BEST_PRACTICES = "best_practices"


class TutorialFormat(Enum):
    """Tutorial content formats."""
    MARKDOWN = "markdown"
    JUPYTER = "jupyter"
    INTERACTIVE = "interactive"
    VIDEO = "video"
    MIXED = "mixed"


@dataclass
class TutorialCard:
    """Represents a tutorial card in the grid."""
    title: str
    description: str
    category: TutorialCategory
    difficulty: TutorialDifficulty
    format: TutorialFormat
    thumbnail_url: str = ""
    duration_minutes: int = 15
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)
    file_path: str = ""
    interactive_elements: List[str] = field(default_factory=list)
    completion_rate: float = 0.0
    user_rating: float = 0.0


@dataclass
class TutorialSequence:
    """Represents a sequence of related tutorials."""
    name: str
    description: str
    cards: List[TutorialCard] = field(default_factory=list)
    estimated_time: int = 0
    completion_requirements: List[str] = field(default_factory=list)
    certificate_available: bool = False


@dataclass
class GridLayout:
    """Defines the visual grid layout."""
    columns: int = 3
    cards_per_page: int = 12
    categories_visible: List[TutorialCategory] = field(default_factory=list)
    difficulty_filters: List[TutorialDifficulty] = field(default_factory=list)
    sort_by: str = "difficulty"  # difficulty, duration, popularity
    theme: str = "default"


class TutorialGridSystem:
    """
    Tutorial grid system inspired by AutoGen's visual card-based navigation
    with scenario-based examples and graduated complexity.
    """
    
    def __init__(self, content_dir: str = "tutorials"):
        """Initialize tutorial grid system."""
        self.content_dir = Path(content_dir)
        self.cards = []
        self.sequences = []
        self.grid_layout = GridLayout()
        self.analytics = {"views": {}, "completions": {}, "ratings": {}}
        logger.info(f"Tutorial grid system initialized at {content_dir}")
        
    def create_tutorial_card(self, 
                           title: str,
                           description: str,
                           category: TutorialCategory,
                           difficulty: TutorialDifficulty,
                           **kwargs) -> TutorialCard:
        """Create a new tutorial card."""
        card = TutorialCard(
            title=title,
            description=description,
            category=category,
            difficulty=difficulty,
            **kwargs
        )
        
        self.cards.append(card)
        logger.info(f"Created tutorial card: {title}")
        return card
        
    def create_scenario_based_tutorial(self,
                                     scenario_name: str,
                                     description: str,
                                     real_world_context: str,
                                     steps: List[Dict[str, Any]]) -> TutorialCard:
        """Create scenario-based tutorial card."""
        # Generate code examples from steps
        code_examples = []
        for step in steps:
            if 'code' in step:
                code_examples.append(step['code'])
                
        # Determine difficulty based on steps complexity
        difficulty = TutorialDifficulty.BEGINNER
        if len(steps) > 5:
            difficulty = TutorialDifficulty.INTERMEDIATE
        if any('advanced' in str(step).lower() for step in steps):
            difficulty = TutorialDifficulty.ADVANCED
            
        card = TutorialCard(
            title=f"Scenario: {scenario_name}",
            description=f"{description}\n\n**Context:** {real_world_context}",
            category=TutorialCategory.REAL_WORLD,
            difficulty=difficulty,
            format=TutorialFormat.MIXED,
            scenarios=[scenario_name],
            code_examples=code_examples,
            duration_minutes=len(steps) * 3,  # Estimate 3 min per step
            learning_objectives=[f"Complete {scenario_name} scenario"]
        )
        
        self.cards.append(card)
        logger.info(f"Created scenario tutorial: {scenario_name}")
        return card
        
    def create_progressive_sequence(self, 
                                  sequence_name: str,
                                  cards: List[TutorialCard],
                                  learning_path: List[str]) -> TutorialSequence:
        """Create progressive learning sequence."""
        # Sort cards by difficulty for progression
        sorted_cards = sorted(cards, key=lambda x: list(TutorialDifficulty).index(x.difficulty))
        
        # Calculate total time
        total_time = sum(card.duration_minutes for card in sorted_cards)
        
        sequence = TutorialSequence(
            name=sequence_name,
            description=f"Progressive learning path: {' → '.join(learning_path)}",
            cards=sorted_cards,
            estimated_time=total_time,
            completion_requirements=learning_path
        )
        
        self.sequences.append(sequence)
        logger.info(f"Created tutorial sequence: {sequence_name}")
        return sequence
        
    def generate_grid_html(self, layout: Optional[GridLayout] = None) -> str:
        """Generate HTML grid layout."""
        if not layout:
            layout = self.grid_layout
            
        # Filter cards based on layout settings
        filtered_cards = self._filter_cards(layout)
        
        html = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "    <title>Tutorial Grid</title>",
            "    <style>",
            self._generate_grid_css(layout),
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='tutorial-grid-container'>",
            "        <header class='grid-header'>",
            "            <h1>Interactive Tutorials</h1>",
            "            <div class='filters'>",
            self._generate_filter_controls(),
            "            </div>",
            "        </header>",
            "        <div class='tutorial-grid'>",
        ]
        
        # Generate cards
        for card in filtered_cards:
            html.append(self._generate_card_html(card))
            
        html.extend([
            "        </div>",
            "    </div>",
            "    <script>",
            self._generate_grid_javascript(),
            "    </script>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html)
        
    def _generate_card_html(self, card: TutorialCard) -> str:
        """Generate HTML for tutorial card."""
        difficulty_color = {
            TutorialDifficulty.BEGINNER: "#4CAF50",
            TutorialDifficulty.INTERMEDIATE: "#FF9800", 
            TutorialDifficulty.ADVANCED: "#F44336",
            TutorialDifficulty.EXPERT: "#9C27B0"
        }
        
        card_html = f"""
        <div class="tutorial-card" data-category="{card.category.value}" 
             data-difficulty="{card.difficulty.value}" data-duration="{card.duration_minutes}">
            <div class="card-thumbnail">
                <img src="{card.thumbnail_url or '/static/default-tutorial.png'}" 
                     alt="{card.title}" loading="lazy">
                <div class="difficulty-badge" 
                     style="background-color: {difficulty_color[card.difficulty]}">
                    {card.difficulty.value.title()}
                </div>
            </div>
            <div class="card-content">
                <h3 class="card-title">{card.title}</h3>
                <p class="card-description">{card.description[:120]}...</p>
                <div class="card-meta">
                    <span class="duration">⏱️ {card.duration_minutes} min</span>
                    <span class="format">📄 {card.format.value}</span>
                </div>
                <div class="card-tags">
        """
        
        # Add tags
        for tag in card.tags[:3]:  # Show max 3 tags
            card_html += f'<span class="tag">{tag}</span>'
            
        # Add scenarios if any
        for scenario in card.scenarios[:2]:  # Show max 2 scenarios
            card_html += f'<span class="scenario">🎯 {scenario}</span>'
            
        card_html += f"""
                </div>
                <div class="card-actions">
                    <button class="btn-primary" onclick="startTutorial('{card.file_path}')">
                        Start Tutorial
                    </button>
                    <button class="btn-secondary" onclick="previewTutorial('{card.title}')">
                        Preview
                    </button>
                </div>
            </div>
        </div>
        """
        
        return card_html
        
    def _generate_grid_css(self, layout: GridLayout) -> str:
        """Generate CSS for grid layout."""
        return f"""
        .tutorial-grid-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        .grid-header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .grid-header h1 {{
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .filters {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .tutorial-grid {{
            display: grid;
            grid-template-columns: repeat({layout.columns}, 1fr);
            gap: 24px;
            margin-top: 30px;
        }}
        
        @media (max-width: 768px) {{
            .tutorial-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media (max-width: 1024px) and (min-width: 769px) {{
            .tutorial-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        .tutorial-card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }}
        
        .tutorial-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .card-thumbnail {{
            position: relative;
            height: 180px;
            overflow: hidden;
        }}
        
        .card-thumbnail img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .difficulty-badge {{
            position: absolute;
            top: 10px;
            right: 10px;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .card-content {{
            padding: 20px;
        }}
        
        .card-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 0 0 10px 0;
            line-height: 1.3;
        }}
        
        .card-description {{
            color: #666;
            line-height: 1.5;
            margin-bottom: 15px;
            font-size: 0.9rem;
        }}
        
        .card-meta {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            font-size: 0.85rem;
            color: #888;
        }}
        
        .card-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 20px;
        }}
        
        .tag, .scenario {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .scenario {{
            background: #fff3e0;
            color: #f57c00;
        }}
        
        .card-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .btn-primary {{
            flex: 1;
            background: #3b82f6;
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }}
        
        .btn-primary:hover {{
            background: #2563eb;
        }}
        
        .btn-secondary {{
            background: white;
            color: #6b7280;
            border: 1px solid #d1d5db;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }}
        
        .btn-secondary:hover {{
            background: #f9fafb;
            border-color: #9ca3af;
        }}
        """
        
    def _generate_filter_controls(self) -> str:
        """Generate filter control HTML."""
        controls = []
        
        # Category filter
        controls.append("""
            <div class="filter-group">
                <label for="category-filter">Category:</label>
                <select id="category-filter">
                    <option value="all">All Categories</option>
        """)
        
        for category in TutorialCategory:
            controls.append(f'<option value="{category.value}">{category.value.replace("_", " ").title()}</option>')
            
        controls.append("</select></div>")
        
        # Difficulty filter
        controls.append("""
            <div class="filter-group">
                <label for="difficulty-filter">Difficulty:</label>
                <select id="difficulty-filter">
                    <option value="all">All Levels</option>
        """)
        
        for difficulty in TutorialDifficulty:
            controls.append(f'<option value="{difficulty.value}">{difficulty.value.title()}</option>')
            
        controls.append("</select></div>")
        
        # Duration filter
        controls.append("""
            <div class="filter-group">
                <label for="duration-filter">Duration:</label>
                <select id="duration-filter">
                    <option value="all">Any Duration</option>
                    <option value="short">< 15 min</option>
                    <option value="medium">15-30 min</option>
                    <option value="long">> 30 min</option>
                </select>
            </div>
        """)
        
        return "\n".join(controls)
        
    def _generate_grid_javascript(self) -> str:
        """Generate JavaScript for grid interactivity."""
        return """
        // Tutorial grid functionality
        function filterTutorials() {
            const categoryFilter = document.getElementById('category-filter').value;
            const difficultyFilter = document.getElementById('difficulty-filter').value;
            const durationFilter = document.getElementById('duration-filter').value;
            
            const cards = document.querySelectorAll('.tutorial-card');
            
            cards.forEach(card => {
                const category = card.dataset.category;
                const difficulty = card.dataset.difficulty;
                const duration = parseInt(card.dataset.duration);
                
                let show = true;
                
                if (categoryFilter !== 'all' && category !== categoryFilter) {
                    show = false;
                }
                
                if (difficultyFilter !== 'all' && difficulty !== difficultyFilter) {
                    show = false;
                }
                
                if (durationFilter !== 'all') {
                    if (durationFilter === 'short' && duration >= 15) show = false;
                    if (durationFilter === 'medium' && (duration < 15 || duration > 30)) show = false;
                    if (durationFilter === 'long' && duration <= 30) show = false;
                }
                
                card.style.display = show ? 'block' : 'none';
            });
        }
        
        function startTutorial(filePath) {
            if (filePath) {
                window.open(filePath, '_blank');
            } else {
                alert('Tutorial not available yet!');
            }
        }
        
        function previewTutorial(title) {
            alert(`Preview for: ${title}\\n\\nThis would show a quick overview of the tutorial content.`);
        }
        
        // Add event listeners
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('category-filter').addEventListener('change', filterTutorials);
            document.getElementById('difficulty-filter').addEventListener('change', filterTutorials);
            document.getElementById('duration-filter').addEventListener('change', filterTutorials);
            
            // Add card click handlers
            document.querySelectorAll('.tutorial-card').forEach(card => {
                card.addEventListener('click', function(e) {
                    if (!e.target.closest('.card-actions')) {
                        const title = this.querySelector('.card-title').textContent;
                        previewTutorial(title);
                    }
                });
            });
        });
        
        // Search functionality
        function searchTutorials(query) {
            const cards = document.querySelectorAll('.tutorial-card');
            const searchTerm = query.toLowerCase();
            
            cards.forEach(card => {
                const title = card.querySelector('.card-title').textContent.toLowerCase();
                const description = card.querySelector('.card-description').textContent.toLowerCase();
                const tags = Array.from(card.querySelectorAll('.tag, .scenario'))
                    .map(tag => tag.textContent.toLowerCase()).join(' ');
                
                const matches = title.includes(searchTerm) || 
                               description.includes(searchTerm) || 
                               tags.includes(searchTerm);
                
                card.style.display = matches ? 'block' : 'none';
            });
        }
        """
        
    def _filter_cards(self, layout: GridLayout) -> List[TutorialCard]:
        """Filter cards based on layout settings."""
        filtered = self.cards
        
        # Filter by categories
        if layout.categories_visible:
            filtered = [card for card in filtered if card.category in layout.categories_visible]
            
        # Filter by difficulties
        if layout.difficulty_filters:
            filtered = [card for card in filtered if card.difficulty in layout.difficulty_filters]
            
        # Sort cards
        if layout.sort_by == "difficulty":
            filtered = sorted(filtered, key=lambda x: list(TutorialDifficulty).index(x.difficulty))
        elif layout.sort_by == "duration":
            filtered = sorted(filtered, key=lambda x: x.duration_minutes)
        elif layout.sort_by == "popularity":
            filtered = sorted(filtered, key=lambda x: x.user_rating, reverse=True)
            
        return filtered
        
    def generate_learning_path_json(self, sequence: TutorialSequence) -> str:
        """Generate JSON for learning path configuration."""
        path_config = {
            "sequence_name": sequence.name,
            "description": sequence.description,
            "estimated_time_minutes": sequence.estimated_time,
            "certificate_available": sequence.certificate_available,
            "steps": []
        }
        
        for i, card in enumerate(sequence.cards):
            step = {
                "step_number": i + 1,
                "title": card.title,
                "description": card.description,
                "difficulty": card.difficulty.value,
                "duration_minutes": card.duration_minutes,
                "prerequisites": card.prerequisites,
                "learning_objectives": card.learning_objectives,
                "file_path": card.file_path,
                "interactive_elements": card.interactive_elements
            }
            path_config["steps"].append(step)
            
        return json.dumps(path_config, indent=2)
        
    def create_real_world_scenarios(self) -> List[TutorialCard]:
        """Create common real-world scenario tutorials."""
        scenarios = [
            {
                "name": "Travel Planning Assistant",
                "description": "Build an AI agent that helps plan complete travel itineraries",
                "context": "Tourism and hospitality industry needs automated trip planning",
                "steps": [
                    {"step": "Set up destination research agent", "code": "agent = TravelAgent(research_mode=True)"},
                    {"step": "Integrate weather and events APIs", "code": "agent.add_data_sources(['weather', 'events'])"},
                    {"step": "Create itinerary generation logic", "code": "itinerary = agent.generate_itinerary(dates, preferences)"},
                    {"step": "Add booking integration", "code": "agent.connect_booking_apis()"},
                    {"step": "Implement user feedback loop", "code": "agent.refine_suggestions(user_feedback)"}
                ]
            },
            {
                "name": "Company Research Agent",  
                "description": "Create an agent that researches companies for investment decisions",
                "context": "Financial services need automated company analysis",
                "steps": [
                    {"step": "Set up data collection agent", "code": "researcher = CompanyResearcher()"},
                    {"step": "Connect financial data sources", "code": "researcher.add_sources(['sec_filings', 'market_data'])"},
                    {"step": "Implement analysis workflows", "code": "analysis = researcher.analyze_company(ticker)"},
                    {"step": "Generate investment reports", "code": "report = researcher.generate_report(analysis)"},
                    {"step": "Add risk assessment", "code": "risks = researcher.assess_risks(company_data)"}
                ]
            },
            {
                "name": "Literature Review Assistant",
                "description": "Build an agent that conducts systematic literature reviews", 
                "context": "Academic researchers need automated paper analysis",
                "steps": [
                    {"step": "Set up paper search agent", "code": "reviewer = LiteratureReviewer()"},
                    {"step": "Connect academic databases", "code": "reviewer.connect_databases(['pubmed', 'arxiv'])"},
                    {"step": "Implement relevance scoring", "code": "scores = reviewer.score_relevance(papers, query)"},
                    {"step": "Extract key findings", "code": "findings = reviewer.extract_findings(relevant_papers)"},
                    {"step": "Generate review summary", "code": "summary = reviewer.synthesize_findings(findings)"}
                ]
            }
        ]
        
        cards = []
        for scenario_data in scenarios:
            card = self.create_scenario_based_tutorial(
                scenario_data["name"],
                scenario_data["description"],
                scenario_data["context"], 
                scenario_data["steps"]
            )
            cards.append(card)
            
        return cards
        
    def export_grid_config(self, output_path: str) -> None:
        """Export complete grid configuration."""
        config = {
            "grid_layout": {
                "columns": self.grid_layout.columns,
                "cards_per_page": self.grid_layout.cards_per_page,
                "sort_by": self.grid_layout.sort_by,
                "theme": self.grid_layout.theme
            },
            "tutorial_cards": [
                {
                    "title": card.title,
                    "description": card.description,
                    "category": card.category.value,
                    "difficulty": card.difficulty.value,
                    "format": card.format.value,
                    "duration_minutes": card.duration_minutes,
                    "prerequisites": card.prerequisites,
                    "learning_objectives": card.learning_objectives,
                    "tags": card.tags,
                    "scenarios": card.scenarios,
                    "file_path": card.file_path
                }
                for card in self.cards
            ],
            "tutorial_sequences": [
                {
                    "name": seq.name,
                    "description": seq.description,
                    "card_count": len(seq.cards),
                    "estimated_time": seq.estimated_time,
                    "completion_requirements": seq.completion_requirements
                }
                for seq in self.sequences
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Exported grid config to {output_path}")