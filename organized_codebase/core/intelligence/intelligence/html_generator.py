"""
HTML Dashboard Generator Module
Generates interactive HTML dashboards for test quality metrics.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json


class HTMLGenerator:
    """Generates HTML dashboards from metrics data."""
    
    def __init__(self, template_dir: Path = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        
    def generate_dashboard(self, data: Dict[str, Any], output_path: Path) -> Path:
        """Generate complete HTML dashboard."""
        html_content = self._build_html(data)
        output_path.write_text(html_content, encoding='utf-8')
        return output_path
    
    def _build_html(self, data: Dict[str, Any]) -> str:
        """Build complete HTML content."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    {self._generate_head()}
</head>
<body>
    <div class="container">
        {self._generate_header(data)}
        {self._generate_metrics_grid(data)}
        {self._generate_recommendations(data.get('recommendations', []))}
        {self._generate_heatmap(data.get('heatmap_data', {}))}
        {self._generate_module_table(data.get('modules', []))}
        {self._generate_test_table(data.get('tests', []))}
        {self._generate_charts(data)}
        {self._generate_footer(data)}
    </div>
    {self._generate_scripts(data)}
</body>
</html>
"""
    
    def _generate_head(self) -> str:
        """Generate HTML head section."""
        return """
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Quality Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        /* Header Styles */
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 40px; 
            border-radius: 15px; 
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.95; font-size: 1.1em; }
        
        /* Metrics Grid */
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .metric-card { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .metric-card h3 { 
            color: #666; 
            font-size: 0.9em; 
            margin-bottom: 10px; 
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card .value { 
            font-size: 2.2em; 
            font-weight: bold; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card .trend { 
            font-size: 0.9em; 
            color: #666; 
            margin-top: 8px;
            display: flex;
            align-items: center;
        }
        
        /* Recommendations */
        .recommendations { 
            background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
            border-left: 5px solid #ffc107; 
            padding: 25px; 
            border-radius: 10px; 
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .recommendations h2 { color: #856404; margin-bottom: 15px; }
        .recommendations ul { list-style: none; }
        .recommendations li { 
            color: #856404; 
            margin-bottom: 10px;
            padding-left: 25px;
            position: relative;
        }
        .recommendations li:before {
            content: "→";
            position: absolute;
            left: 0;
            font-weight: bold;
        }
        
        /* Heatmap */
        .heatmap-container {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .heatmap { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); 
            gap: 8px; 
            padding: 15px;
        }
        .heatmap-cell { 
            padding: 20px 10px; 
            text-align: center; 
            border-radius: 8px; 
            color: white; 
            font-size: 0.9em;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        .heatmap-cell:hover { 
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* Tables */
        .table-container { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px; 
            overflow-x: auto; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
        }
        th { 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 15px; 
            text-align: left; 
            border-bottom: 2px solid #dee2e6;
            font-weight: 600;
            color: #495057;
        }
        td { 
            padding: 12px 15px; 
            border-bottom: 1px solid #dee2e6; 
        }
        tr:hover { 
            background: #f8f9fa; 
        }
        
        /* Status Colors */
        .status-passed { color: #28a745; font-weight: 600; }
        .status-failed { color: #dc3545; font-weight: 600; }
        .status-skipped { color: #ffc107; font-weight: 600; }
        
        /* Charts */
        .chart-container { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px; 
        }
        
        /* Footer */
        .footer { 
            text-align: center; 
            color: #666; 
            margin-top: 50px; 
            padding: 30px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .container > * {
            animation: fadeIn 0.5s ease-out;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
"""
    
    def _generate_header(self, data: Dict[str, Any]) -> str:
        """Generate header section."""
        generated_at = data.get('generated_at', datetime.now())
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at)
        
        return f"""
        <div class="header">
            <h1>🚀 TestMaster Quality Dashboard</h1>
            <p>Generated on {generated_at.strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Comprehensive test quality metrics and insights</p>
        </div>
"""
    
    def _generate_metrics_grid(self, data: Dict[str, Any]) -> str:
        """Generate metrics grid."""
        metrics = [
            ("Total Tests", data.get('total_tests', 0), "📊"),
            ("Coverage", f"{data.get('overall_coverage', 0):.1f}%", "🎯"),
            ("Quality Score", f"{data.get('average_quality', 0):.1f}", "⭐"),
            ("Total Modules", data.get('total_modules', 0), "📦"),
            ("Exec Time", f"{data.get('test_execution_time', 0):.2f}s", "⏱️"),
            ("Pass Rate", f"{self._calculate_pass_rate(data):.1f}%", "✅")
        ]
        
        cards = []
        for title, value, icon in metrics:
            cards.append(f"""
            <div class="metric-card">
                <h3>{icon} {title}</h3>
                <div class="value">{value}</div>
                <div class="trend">{self._get_trend_indicator(title, data)}</div>
            </div>
            """)
        
        return f'<div class="metrics-grid">{"".join(cards)}</div>'
    
    def _generate_recommendations(self, recommendations: List[str]) -> str:
        """Generate recommendations section."""
        if not recommendations:
            return ""
        
        items = "".join(f"<li>{rec}</li>" for rec in recommendations)
        
        return f"""
        <div class="recommendations">
            <h2>📝 Recommendations</h2>
            <ul>{items}</ul>
        </div>
"""
    
    def _generate_heatmap(self, heatmap_data: Dict[str, Any]) -> str:
        """Generate coverage heatmap."""
        if not heatmap_data or 'modules' not in heatmap_data:
            return ""
        
        cells = []
        for module in heatmap_data['modules']:
            color = module.get('color', '#e74c3c')
            cells.append(f"""
            <div class="heatmap-cell" style="background: {color};" 
                 title="{module['name']}: {module['coverage']:.1f}% coverage">
                <div>{module['name'][:15]}</div>
                <strong>{module['coverage']:.0f}%</strong>
            </div>
            """)
        
        return f"""
        <div class="heatmap-container">
            <h2>📊 Coverage Heatmap</h2>
            <div class="heatmap">{"".join(cells)}</div>
        </div>
"""
    
    def _generate_module_table(self, modules: List[Dict]) -> str:
        """Generate modules table."""
        if not modules:
            return ""
        
        rows = []
        for module in modules[:10]:  # Show top 10
            trend_icon = "📈" if module.get('trend') == 'improving' else "📉" if module.get('trend') == 'declining' else "➡️"
            rows.append(f"""
            <tr>
                <td>{module['name']}</td>
                <td>{module['total_tests']}</td>
                <td>{module['coverage_percentage']:.1f}%</td>
                <td>{module['average_quality']:.1f}</td>
                <td>{trend_icon} {module['trend']}</td>
            </tr>
            """)
        
        return f"""
        <div class="table-container">
            <h2>📦 Module Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Module</th>
                        <th>Tests</th>
                        <th>Coverage</th>
                        <th>Quality</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>
"""
    
    def _generate_test_table(self, tests: List[Dict]) -> str:
        """Generate tests table."""
        # Implementation continues in next module...
        return ""
    
    def _generate_charts(self, data: Dict[str, Any]) -> str:
        """Generate chart sections."""
        return ""  # Charts handled by JavaScript
    
    def _generate_footer(self, data: Dict[str, Any]) -> str:
        """Generate footer."""
        return """
        <div class="footer">
            <p>TestMaster Quality Dashboard v2.0</p>
            <p>© 2025 TestMaster - Intelligent Test Framework</p>
        </div>
"""
    
    def _generate_scripts(self, data: Dict[str, Any]) -> str:
        """Generate JavaScript for interactivity."""
        return f"""
    <script>
        const dashboardData = {json.dumps(data, default=str)};
        console.log('Dashboard loaded with', Object.keys(dashboardData));
    </script>
"""
    
    def _calculate_pass_rate(self, data: Dict[str, Any]) -> float:
        """Calculate overall pass rate."""
        modules = data.get('modules', [])
        if not modules:
            return 0.0
        
        total_passed = sum(m.get('passed_tests', 0) for m in modules)
        total_tests = sum(m.get('total_tests', 0) for m in modules)
        
        return (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    def _get_trend_indicator(self, metric: str, data: Dict[str, Any]) -> str:
        """Get trend indicator for a metric."""
        trends = data.get('trends', {})
        trend = trends.get(metric.lower().replace(" ", "_"), "stable")
        
        if trend == "improving":
            return "📈 Improving"
        elif trend == "declining":
            return "📉 Declining"
        else:
            return "➡️ Stable"