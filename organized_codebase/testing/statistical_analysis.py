"""
Statistical Analysis Module
===========================

Implements comprehensive statistical analysis:
- Distribution analysis and normality testing
- Correlation analysis between metrics
- Outlier detection and anomaly identification
- Clustering and pattern recognition
- Trend analysis and forecasting
- Information theory measures
"""

import ast
import math
import statistics
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import itertools

from .base_analyzer import BaseAnalyzer


class StatisticalAnalyzer(BaseAnalyzer):
    """Analyzer for statistical patterns in code metrics."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        print("[INFO] Analyzing Statistical Patterns...")
        
        # First collect all metrics from the codebase
        metrics_data = self._collect_code_metrics()
        
        results = {
            "distribution_analysis": self._analyze_distributions(metrics_data),
            "correlation_analysis": self._analyze_correlations(metrics_data),
            "outlier_detection": self._detect_outliers(metrics_data),
            "clustering_analysis": self._perform_clustering(metrics_data),
            "trend_analysis": self._analyze_trends(metrics_data),
            "variance_analysis": self._analyze_variance(metrics_data),
            "entropy_measures": self._calculate_entropy_measures(metrics_data),
            "information_theory": self._apply_information_theory(metrics_data),
            "statistical_tests": self._perform_statistical_tests(metrics_data)
        }
        
        print(f"  [OK] Analyzed {len(results)} statistical categories")
        return results
    
    def _collect_code_metrics(self) -> Dict[str, List[float]]:
        """Collect various code metrics for statistical analysis."""
        metrics = {
            'file_sizes': [],
            'function_lengths': [],
            'cyclomatic_complexity': [],
            'nesting_depth': [],
            'identifier_length': [],
            'comment_ratio': [],
            'import_count': [],
            'class_method_count': [],
            'lines_per_function': [],
            'parameters_per_function': []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                lines = content.split('\n')
                
                # File-level metrics
                file_size = len(content)
                metrics['file_sizes'].append(file_size)
                
                # Count imports
                import_count = 0
                comment_lines = 0
                
                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        import_count += 1
                    if line.strip().startswith('#'):
                        comment_lines += 1
                
                metrics['import_count'].append(import_count)
                metrics['comment_ratio'].append(comment_lines / max(len(lines), 1))
                
                # Function and class metrics
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Function complexity
                        complexity = self._calculate_function_complexity(node)
                        metrics['cyclomatic_complexity'].append(complexity)
                        
                        # Function length
                        func_end = node.end_lineno or (node.lineno + 10)
                        func_length = func_end - node.lineno + 1
                        metrics['function_lengths'].append(func_length)
                        metrics['lines_per_function'].append(func_length)
                        
                        # Parameter count
                        param_count = len(node.args.args)
                        metrics['parameters_per_function'].append(param_count)
                        
                        # Nesting depth
                        max_depth = self._calculate_max_nesting_depth(node)
                        metrics['nesting_depth'].append(max_depth)
                        
                        # Identifier length
                        metrics['identifier_length'].append(len(node.name))
                    
                    elif isinstance(node, ast.ClassDef):
                        # Class method count
                        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        metrics['class_method_count'].append(method_count)
                        
                        # Class name length
                        metrics['identifier_length'].append(len(node.name))
                    
                    elif isinstance(node, ast.Name):
                        # Variable name lengths
                        metrics['identifier_length'].append(len(node.id))
                        
            except Exception:
                continue
        
        return metrics
    
    def _calculate_max_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth recursively."""
        max_depth = current_depth
        
        nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                child_depth = self._calculate_max_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _analyze_distributions(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze statistical distributions of metrics."""
        distribution_analysis = {}
        
        for metric_name, values in metrics_data.items():
            if not values:
                continue
            
            # Basic statistics
            analysis = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values)
            }
            
            # Percentiles
            sorted_values = sorted(values)
            n = len(sorted_values)
            analysis['percentiles'] = {
                '25th': sorted_values[int(0.25 * n)],
                '75th': sorted_values[int(0.75 * n)],
                '90th': sorted_values[int(0.90 * n)],
                '95th': sorted_values[int(0.95 * n)]
            }
            
            # Skewness (simplified calculation)
            if analysis['std_dev'] > 0:
                skewness_sum = sum((x - analysis['mean'])**3 for x in values)
                analysis['skewness'] = skewness_sum / (len(values) * analysis['std_dev']**3)
            else:
                analysis['skewness'] = 0
            
            # Kurtosis (simplified calculation)
            if analysis['std_dev'] > 0:
                kurtosis_sum = sum((x - analysis['mean'])**4 for x in values)
                analysis['kurtosis'] = (kurtosis_sum / (len(values) * analysis['std_dev']**4)) - 3
            else:
                analysis['kurtosis'] = 0
            
            # Distribution shape classification
            analysis['distribution_shape'] = self._classify_distribution_shape(
                analysis['skewness'], analysis['kurtosis']
            )
            
            # Normality assessment (basic)
            analysis['appears_normal'] = (
                abs(analysis['skewness']) < 1.0 and 
                abs(analysis['kurtosis']) < 3.0
            )
            
            # Coefficient of variation
            if analysis['mean'] != 0:
                analysis['coefficient_of_variation'] = analysis['std_dev'] / abs(analysis['mean'])
            else:
                analysis['coefficient_of_variation'] = float('inf')
            
            distribution_analysis[metric_name] = analysis
        
        return distribution_analysis
    
    def _classify_distribution_shape(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution shape based on skewness and kurtosis."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "approximately_normal"
        elif skewness > 1.0:
            return "highly_right_skewed"
        elif skewness > 0.5:
            return "moderately_right_skewed"
        elif skewness < -1.0:
            return "highly_left_skewed"
        elif skewness < -0.5:
            return "moderately_left_skewed"
        elif kurtosis > 3.0:
            return "heavy_tailed"
        elif kurtosis < -1.5:
            return "light_tailed"
        else:
            return "symmetric"
    
    def _analyze_correlations(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        correlations = {}
        metric_pairs = []
        
        # Filter metrics with sufficient data
        valid_metrics = {k: v for k, v in metrics_data.items() if len(v) > 10}
        
        # Calculate pairwise correlations
        for metric1, metric2 in itertools.combinations(valid_metrics.keys(), 2):
            values1 = valid_metrics[metric1]
            values2 = valid_metrics[metric2]
            
            # Ensure both metrics have the same number of data points
            min_len = min(len(values1), len(values2))
            if min_len > 5:
                correlation = self._calculate_pearson_correlation(
                    values1[:min_len], values2[:min_len]
                )
                
                if correlation is not None:
                    metric_pairs.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'strength': self._classify_correlation_strength(correlation),
                        'direction': 'positive' if correlation > 0 else 'negative',
                        'sample_size': min_len
                    })
        
        # Sort by absolute correlation strength
        metric_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Identify strong correlations
        strong_correlations = [pair for pair in metric_pairs if abs(pair['correlation']) > 0.7]
        moderate_correlations = [pair for pair in metric_pairs if 0.3 < abs(pair['correlation']) <= 0.7]
        
        return {
            'all_correlations': metric_pairs,
            'strong_correlations': strong_correlations,
            'moderate_correlations': moderate_correlations,
            'correlation_summary': {
                'total_pairs_analyzed': len(metric_pairs),
                'strong_correlations_count': len(strong_correlations),
                'moderate_correlations_count': len(moderate_correlations),
                'highest_positive_correlation': max(
                    [p['correlation'] for p in metric_pairs if p['correlation'] > 0], default=0
                ),
                'highest_negative_correlation': min(
                    [p['correlation'] for p in metric_pairs if p['correlation'] < 0], default=0
                )
            }
        }
    
    def _calculate_pearson_correlation(self, values1: List[float], values2: List[float]) -> Optional[float]:
        """Calculate Pearson correlation coefficient."""
        if len(values1) != len(values2) or len(values1) < 2:
            return None
        
        try:
            n = len(values1)
            sum1 = sum(values1)
            sum2 = sum(values2)
            sum1_sq = sum(x**2 for x in values1)
            sum2_sq = sum(x**2 for x in values2)
            sum_products = sum(x*y for x, y in zip(values1, values2))
            
            numerator = n * sum_products - sum1 * sum2
            denominator = math.sqrt((n * sum1_sq - sum1**2) * (n * sum2_sq - sum2**2))
            
            if denominator == 0:
                return 0
            
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            return None
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr > 0.9:
            return "very_strong"
        elif abs_corr > 0.7:
            return "strong"
        elif abs_corr > 0.5:
            return "moderate"
        elif abs_corr > 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _detect_outliers(self, metrics_data: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect outliers in metrics using IQR and Z-score methods."""
        outliers = []
        outlier_id = 1
        
        for metric_name, values in metrics_data.items():
            if len(values) < 10:  # Need sufficient data
                continue
            
            # IQR method
            sorted_values = sorted(values)
            n = len(sorted_values)
            q1 = sorted_values[int(0.25 * n)]
            q3 = sorted_values[int(0.75 * n)]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Z-score method
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            for i, value in enumerate(values):
                is_outlier_iqr = value < lower_bound or value > upper_bound
                
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                is_outlier_zscore = z_score > 3  # 3-sigma rule
                
                if is_outlier_iqr or is_outlier_zscore:
                    outliers.append({
                        'outlier_id': outlier_id,
                        'metric': metric_name,
                        'value': value,
                        'index': i,
                        'z_score': z_score,
                        'deviation_from_mean': abs(value - mean_val),
                        'method_detected': 'both' if (is_outlier_iqr and is_outlier_zscore) 
                                          else 'iqr' if is_outlier_iqr else 'zscore',
                        'severity': self._classify_outlier_severity(z_score),
                        'percentile': self._calculate_percentile(value, sorted_values)
                    })
                    outlier_id += 1
        
        # Sort by severity
        outliers.sort(key=lambda x: x['z_score'], reverse=True)
        
        return outliers
    
    def _classify_outlier_severity(self, z_score: float) -> str:
        """Classify outlier severity based on Z-score."""
        if z_score > 5:
            return "extreme"
        elif z_score > 3:
            return "severe"
        elif z_score > 2:
            return "moderate"
        else:
            return "mild"
    
    def _calculate_percentile(self, value: float, sorted_values: List[float]) -> float:
        """Calculate percentile of a value in a sorted list."""
        if not sorted_values:
            return 0
        
        # Find position of value
        count_below = sum(1 for v in sorted_values if v < value)
        return (count_below / len(sorted_values)) * 100
    
    def _perform_clustering(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform simple clustering analysis on metrics."""
        # Simplified k-means clustering implementation
        clustering_results = {}
        
        for metric_name, values in metrics_data.items():
            if len(values) < 20:  # Need sufficient data for clustering
                continue
            
            # Simple k-means with k=3
            clusters = self._simple_kmeans(values, k=3)
            
            if clusters:
                clustering_results[metric_name] = {
                    'cluster_count': len(clusters),
                    'clusters': clusters,
                    'cluster_quality': self._evaluate_cluster_quality(clusters),
                    'within_cluster_variance': self._calculate_within_cluster_variance(clusters),
                    'between_cluster_variance': self._calculate_between_cluster_variance(clusters)
                }
        
        return {
            'per_metric_clustering': clustering_results,
            'summary': {
                'metrics_clustered': len(clustering_results),
                'average_cluster_quality': statistics.mean([
                    result['cluster_quality'] for result in clustering_results.values()
                ]) if clustering_results else 0
            }
        }
    
    def _simple_kmeans(self, values: List[float], k: int = 3, max_iterations: int = 10) -> List[Dict[str, Any]]:
        """Simple k-means clustering implementation."""
        if len(values) < k:
            return []
        
        # Initialize centroids
        sorted_values = sorted(values)
        n = len(sorted_values)
        centroids = [sorted_values[i * n // k] for i in range(k)]
        
        for iteration in range(max_iterations):
            # Assign points to clusters
            clusters = [[] for _ in range(k)]
            
            for value in values:
                # Find closest centroid
                distances = [abs(value - centroid) for centroid in centroids]
                closest_cluster = distances.index(min(distances))
                clusters[closest_cluster].append(value)
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroids.append(statistics.mean(cluster))
                else:
                    new_centroids.append(centroids[len(new_centroids)])
            
            # Check for convergence
            if all(abs(old - new) < 0.001 for old, new in zip(centroids, new_centroids)):
                break
            
            centroids = new_centroids
        
        # Format results
        result_clusters = []
        for i, (cluster, centroid) in enumerate(zip(clusters, centroids)):
            if cluster:
                result_clusters.append({
                    'cluster_id': i,
                    'centroid': centroid,
                    'size': len(cluster),
                    'min': min(cluster),
                    'max': max(cluster),
                    'std_dev': statistics.stdev(cluster) if len(cluster) > 1 else 0,
                    'values': cluster[:10]  # Sample of values
                })
        
        return result_clusters
    
    def _evaluate_cluster_quality(self, clusters: List[Dict[str, Any]]) -> float:
        """Evaluate clustering quality using silhouette-like measure."""
        if not clusters or len(clusters) < 2:
            return 0.0
        
        # Simplified quality measure based on within-cluster vs between-cluster distances
        total_within_variance = sum(cluster['std_dev']**2 for cluster in clusters if cluster['size'] > 1)
        
        # Calculate between-cluster separation
        centroids = [cluster['centroid'] for cluster in clusters]
        between_distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                between_distances.append(abs(centroids[i] - centroids[j]))
        
        avg_between_distance = statistics.mean(between_distances) if between_distances else 0
        avg_within_variance = total_within_variance / len(clusters)
        
        # Quality score (higher is better)
        if avg_within_variance > 0:
            return avg_between_distance / avg_within_variance
        else:
            return avg_between_distance
    
    def _calculate_within_cluster_variance(self, clusters: List[Dict[str, Any]]) -> float:
        """Calculate average within-cluster variance."""
        if not clusters:
            return 0.0
        
        total_variance = sum(cluster['std_dev']**2 for cluster in clusters if cluster['size'] > 1)
        return total_variance / len(clusters)
    
    def _calculate_between_cluster_variance(self, clusters: List[Dict[str, Any]]) -> float:
        """Calculate between-cluster variance."""
        if len(clusters) < 2:
            return 0.0
        
        centroids = [cluster['centroid'] for cluster in clusters]
        overall_mean = statistics.mean(centroids)
        
        variance = sum((centroid - overall_mean)**2 for centroid in centroids)
        return variance / len(centroids)
    
    def _analyze_trends(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze trends in metrics (simplified trend detection)."""
        trends = {}
        
        for metric_name, values in metrics_data.items():
            if len(values) < 10:
                continue
            
            # Simple linear trend analysis
            n = len(values)
            x_values = list(range(n))
            
            # Calculate slope using least squares
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean)**2 for x in x_values)
            
            if denominator != 0:
                slope = numerator / denominator
                
                # Calculate R-squared
                y_predicted = [slope * (x - x_mean) + y_mean for x in x_values]
                ss_res = sum((y - y_pred)**2 for y, y_pred in zip(values, y_predicted))
                ss_tot = sum((y - y_mean)**2 for y in values)
                
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                trends[metric_name] = {
                    'slope': slope,
                    'r_squared': r_squared,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'trend_strength': abs(slope),
                    'confidence': r_squared,  # Using R-squared as confidence measure
                    'significant_trend': r_squared > 0.5 and abs(slope) > 0.1
                }
        
        return {
            'per_metric_trends': trends,
            'summary': {
                'metrics_with_trends': len(trends),
                'significant_trends': len([t for t in trends.values() if t['significant_trend']]),
                'increasing_trends': len([t for t in trends.values() if t['trend_direction'] == 'increasing']),
                'decreasing_trends': len([t for t in trends.values() if t['trend_direction'] == 'decreasing'])
            }
        }
    
    def _analyze_variance(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze variance patterns in metrics."""
        variance_analysis = {}
        
        for metric_name, values in metrics_data.items():
            if len(values) < 5:
                continue
            
            variance = statistics.variance(values) if len(values) > 1 else 0
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            mean_val = statistics.mean(values)
            
            # Coefficient of variation
            cv = std_dev / mean_val if mean_val != 0 else float('inf')
            
            variance_analysis[metric_name] = {
                'variance': variance,
                'standard_deviation': std_dev,
                'coefficient_of_variation': cv,
                'variability_level': self._classify_variability(cv),
                'stability_score': 1 / (1 + cv) if cv != float('inf') else 0  # Inverse relationship
            }
        
        # Identify high and low variance metrics
        if variance_analysis:
            variances = [analysis['variance'] for analysis in variance_analysis.values()]
            high_variance_threshold = statistics.mean(variances) + statistics.stdev(variances) if len(variances) > 1 else 0
            
            high_variance_metrics = [
                metric for metric, analysis in variance_analysis.items()
                if analysis['variance'] > high_variance_threshold
            ]
            
            return {
                'per_metric_variance': variance_analysis,
                'summary': {
                    'high_variance_metrics': high_variance_metrics,
                    'high_variance_count': len(high_variance_metrics),
                    'average_cv': statistics.mean([a['coefficient_of_variation'] for a in variance_analysis.values() if a['coefficient_of_variation'] != float('inf')]),
                    'most_stable_metric': min(variance_analysis.items(), key=lambda x: x[1]['coefficient_of_variation'])[0] if variance_analysis else None,
                    'least_stable_metric': max(variance_analysis.items(), key=lambda x: x[1]['coefficient_of_variation'] if x[1]['coefficient_of_variation'] != float('inf') else 0)[0] if variance_analysis else None
                }
            }
        else:
            return {'per_metric_variance': {}, 'summary': {}}
    
    def _classify_variability(self, cv: float) -> str:
        """Classify variability level based on coefficient of variation."""
        if cv == float('inf'):
            return "undefined"
        elif cv < 0.1:
            return "very_low"
        elif cv < 0.3:
            return "low"
        elif cv < 0.6:
            return "moderate"
        elif cv < 1.0:
            return "high"
        else:
            return "very_high"
    
    def _calculate_entropy_measures(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate entropy and information measures."""
        entropy_measures = {}
        
        for metric_name, values in metrics_data.items():
            if len(values) < 10:
                continue
            
            # Discretize values into bins for entropy calculation
            bins = self._create_bins(values, num_bins=10)
            bin_counts = Counter(bins)
            total = len(values)
            
            # Calculate Shannon entropy
            entropy = 0.0
            for count in bin_counts.values():
                probability = count / total
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            # Calculate maximum possible entropy (uniform distribution)
            max_entropy = math.log2(len(bin_counts))
            
            # Normalized entropy (0 to 1)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            entropy_measures[metric_name] = {
                'shannon_entropy': entropy,
                'max_entropy': max_entropy,
                'normalized_entropy': normalized_entropy,
                'information_content': entropy,
                'uniformity': normalized_entropy,  # Higher = more uniform distribution
                'predictability': 1 - normalized_entropy  # Higher = more predictable
            }
        
        return {
            'per_metric_entropy': entropy_measures,
            'summary': {
                'average_entropy': statistics.mean([e['shannon_entropy'] for e in entropy_measures.values()]),
                'average_uniformity': statistics.mean([e['uniformity'] for e in entropy_measures.values()]),
                'most_uniform_metric': max(entropy_measures.items(), key=lambda x: x[1]['uniformity'])[0] if entropy_measures else None,
                'most_predictable_metric': max(entropy_measures.items(), key=lambda x: x[1]['predictability'])[0] if entropy_measures else None
            }
        }
    
    def _create_bins(self, values: List[float], num_bins: int = 10) -> List[int]:
        """Create bins for discretizing continuous values."""
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return [0] * len(values)
        
        bin_width = (max_val - min_val) / num_bins
        bins = []
        
        for value in values:
            bin_index = min(int((value - min_val) / bin_width), num_bins - 1)
            bins.append(bin_index)
        
        return bins
    
    def _apply_information_theory(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Apply information theory concepts to code metrics."""
        info_theory = {}
        
        # Calculate mutual information between metrics (simplified)
        metric_pairs = list(itertools.combinations(metrics_data.items(), 2))
        
        mutual_info_results = []
        
        for (name1, values1), (name2, values2) in metric_pairs:
            if len(values1) > 10 and len(values2) > 10:
                # Simplified mutual information calculation
                min_len = min(len(values1), len(values2))
                mutual_info = self._calculate_mutual_information(
                    values1[:min_len], values2[:min_len]
                )
                
                mutual_info_results.append({
                    'metric1': name1,
                    'metric2': name2,
                    'mutual_information': mutual_info,
                    'information_gain': mutual_info,  # Same in this simplified case
                    'dependency_strength': self._classify_dependency(mutual_info)
                })
        
        # Sort by mutual information
        mutual_info_results.sort(key=lambda x: x['mutual_information'], reverse=True)
        
        return {
            'mutual_information_analysis': mutual_info_results[:10],  # Top 10
            'information_redundancy': self._calculate_information_redundancy(metrics_data),
            'information_density': self._calculate_information_density(metrics_data),
            'summary': {
                'metric_pairs_analyzed': len(mutual_info_results),
                'high_dependency_pairs': len([r for r in mutual_info_results if r['mutual_information'] > 1.0]),
                'information_efficiency': self._calculate_information_efficiency(metrics_data)
            }
        }
    
    def _calculate_mutual_information(self, values1: List[float], values2: List[float]) -> float:
        """Calculate simplified mutual information between two variables."""
        if len(values1) != len(values2) or len(values1) < 10:
            return 0.0
        
        # Discretize both variables
        bins1 = self._create_bins(values1, num_bins=5)
        bins2 = self._create_bins(values2, num_bins=5)
        
        # Create joint distribution
        joint_counts = Counter(zip(bins1, bins2))
        marginal1 = Counter(bins1)
        marginal2 = Counter(bins2)
        
        total = len(values1)
        mutual_info = 0.0
        
        for (bin1, bin2), joint_count in joint_counts.items():
            p_joint = joint_count / total
            p_marginal1 = marginal1[bin1] / total
            p_marginal2 = marginal2[bin2] / total
            
            if p_joint > 0 and p_marginal1 > 0 and p_marginal2 > 0:
                mutual_info += p_joint * math.log2(p_joint / (p_marginal1 * p_marginal2))
        
        return mutual_info
    
    def _classify_dependency(self, mutual_info: float) -> str:
        """Classify dependency strength based on mutual information."""
        if mutual_info > 2.0:
            return "very_strong"
        elif mutual_info > 1.0:
            return "strong"
        elif mutual_info > 0.5:
            return "moderate"
        elif mutual_info > 0.1:
            return "weak"
        else:
            return "very_weak"
    
    def _calculate_information_redundancy(self, metrics_data: Dict[str, List[float]]) -> float:
        """Calculate information redundancy across metrics."""
        # Simplified: based on average pairwise correlations
        correlations = []
        
        for metric1, metric2 in itertools.combinations(metrics_data.keys(), 2):
            values1 = metrics_data[metric1]
            values2 = metrics_data[metric2]
            
            if len(values1) > 10 and len(values2) > 10:
                min_len = min(len(values1), len(values2))
                correlation = self._calculate_pearson_correlation(values1[:min_len], values2[:min_len])
                if correlation is not None:
                    correlations.append(abs(correlation))
        
        return statistics.mean(correlations) if correlations else 0.0
    
    def _calculate_information_density(self, metrics_data: Dict[str, List[float]]) -> float:
        """Calculate information density of the metric set."""
        # Number of unique patterns relative to total data points
        total_points = sum(len(values) for values in metrics_data.values())
        unique_combinations = len(set(
            tuple(values[:10]) for values in metrics_data.values() if len(values) >= 10
        ))
        
        return unique_combinations / max(total_points, 1)
    
    def _calculate_information_efficiency(self, metrics_data: Dict[str, List[float]]) -> float:
        """Calculate information efficiency of the metric set."""
        # Ratio of informative metrics to total metrics
        informative_metrics = len([
            values for values in metrics_data.values() 
            if len(values) > 5 and (max(values) - min(values)) > 0
        ])
        
        return informative_metrics / max(len(metrics_data), 1)
    
    def _perform_statistical_tests(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform basic statistical tests on metrics."""
        test_results = {
            'normality_tests': {},
            'variance_tests': {},
            'mean_comparisons': [],
            'summary_statistics': {}
        }
        
        # Normality tests (simplified Shapiro-Wilk approximation)
        for metric_name, values in metrics_data.items():
            if len(values) > 10:
                normality_score = self._approximate_normality_test(values)
                test_results['normality_tests'][metric_name] = {
                    'normality_score': normality_score,
                    'appears_normal': normality_score > 0.05,  # Using 0.05 as threshold
                    'sample_size': len(values)
                }
        
        # Variance homogeneity tests
        variance_values = [
            statistics.variance(values) for values in metrics_data.values() 
            if len(values) > 5
        ]
        
        if len(variance_values) > 1:
            max_variance = max(variance_values)
            min_variance = min(variance_values)
            variance_ratio = max_variance / max(min_variance, 0.001)
            
            test_results['variance_tests'] = {
                'max_variance': max_variance,
                'min_variance': min_variance,
                'variance_ratio': variance_ratio,
                'homogeneous_variances': variance_ratio < 4.0  # Rule of thumb
            }
        
        # Mean comparison tests (simplified)
        means = [(name, statistics.mean(values)) for name, values in metrics_data.items() if len(values) > 5]
        means.sort(key=lambda x: x[1])
        
        if len(means) > 1:
            test_results['mean_comparisons'] = [
                {
                    'metric': name,
                    'mean': mean_val,
                    'rank': rank + 1
                }
                for rank, (name, mean_val) in enumerate(means)
            ]
        
        return test_results
    
    def _approximate_normality_test(self, values: List[float]) -> float:
        """Approximate normality test (simplified Anderson-Darling)."""
        if len(values) < 8:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate empirical mean and std
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return 0.0
        
        # Normalize values
        normalized = [(x - mean_val) / std_val for x in sorted_values]
        
        # Simple normality approximation based on skewness and kurtosis
        skewness = sum(x**3 for x in normalized) / n
        kurtosis = sum(x**4 for x in normalized) / n - 3
        
        # Combine skewness and kurtosis into a normality score
        normality_score = 1.0 / (1.0 + abs(skewness) + abs(kurtosis))
        
        return normality_score