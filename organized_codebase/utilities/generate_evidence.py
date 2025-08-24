#!/usr/bin/env python3
"""Generate evidence files for architecture dashboard"""

from core.dashboard.architecture_integration_dashboard import get_architecture_dashboard
import json

def main():
    dashboard = get_architecture_dashboard()
    
    # Generate comprehensive metrics
    comprehensive = dashboard._get_comprehensive_metrics()
    
    # Export JSON
    with open('architecture_export_sample.json', 'w') as f:
        json.dump(comprehensive, f, indent=2)
    
    # Generate SVG visualization
    svg_viz = dashboard._generate_svg_visualization(comprehensive)
    with open('architecture_visualization.svg', 'w') as f:
        f.write(svg_viz)
    
    # Generate comprehensive report
    report = dashboard._generate_comprehensive_report()
    with open('architecture_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print('Generated evidence files:')
    print('- architecture_export_sample.json (comprehensive metrics)')
    print('- architecture_visualization.svg (visual representation)')
    print('- architecture_report.json (executive report)')
    print(f'Architecture Health: {comprehensive["performance_summary"]["architecture_health"]:.1f}%')
    print(f'System Services: {comprehensive["system_info"]["total_services"]} services')

if __name__ == '__main__':
    main()