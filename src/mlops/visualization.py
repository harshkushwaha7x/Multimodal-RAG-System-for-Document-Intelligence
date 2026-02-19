"""
Visualization Module.
Generate plots and dashboards for RAG system analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np

from ..utils import get_logger, get_config, LoggerMixin

logger = get_logger(__name__)
config = get_config()


class PlotGenerator(LoggerMixin):
    """
    Generate visualization plots.
    
    Creates:
    - Metric comparison charts
    - Latency distributions
    - Retrieval analysis
    - Embedding visualizations
    """
    
    def __init__(self, output_dir: Optional[Path] = None, style: str = "seaborn"):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Directory for saving plots
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir or config.paths.reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._plt = None
        self._sns = None
    
    def _init_plotting(self):
        """Lazy load plotting libraries."""
        if self._plt is None:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.style.use(self.style)
                sns.set_palette("husl")
                
                self._plt = plt
                self._sns = sns
                
            except ImportError as e:
                self.logger.error(f"Plotting libraries not installed: {e}")
                raise
    
    def metric_comparison_bar(
        self,
        metrics: Dict[str, float],
        title: str = "Metric Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Create bar chart comparing metrics.
        
        Args:
            metrics: Dict of metric names to values
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size
        """
        self._init_plotting()
        
        fig, ax = self._plt.subplots(figsize=figsize)
        
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = self._sns.color_palette("viridis", len(names))
        
        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15)
        
        self._plt.xticks(rotation=45, ha='right')
        self._plt.tight_layout()
        
        if save_path:
            self._plt.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot: {save_path}")
        
        return fig
    
    def latency_distribution(
        self,
        latencies: List[float],
        title: str = "Response Latency Distribution",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Create latency distribution plot with percentile lines.
        
        Args:
            latencies: List of latency values (ms)
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size
        """
        self._init_plotting()
        
        fig, ax = self._plt.subplots(figsize=figsize)
        
        # Histogram with KDE
        self._sns.histplot(latencies, kde=True, color='steelblue', ax=ax, 
                          edgecolor='white', linewidth=0.5)
        
        # Add percentile lines
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        ax.axvline(p50, color='green', linestyle='--', linewidth=2, 
                   label=f'P50: {p50:.0f}ms')
        ax.axvline(p95, color='orange', linestyle='--', linewidth=2,
                   label=f'P95: {p95:.0f}ms')
        ax.axvline(p99, color='red', linestyle='--', linewidth=2,
                   label=f'P99: {p99:.0f}ms')
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        self._plt.tight_layout()
        
        if save_path:
            self._plt.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot: {save_path}")
        
        return fig
    
    def precision_recall_curve(
        self,
        precision_values: List[float],
        recall_values: List[float],
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ):
        """
        Create precision-recall curve.
        
        Args:
            precision_values: Precision at each K
            recall_values: Recall at each K
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size
        """
        self._init_plotting()
        
        fig, ax = self._plt.subplots(figsize=figsize)
        
        ax.plot(recall_values, precision_values, 'b-', linewidth=2, marker='o')
        
        # Add K labels
        for i, (r, p) in enumerate(zip(recall_values, precision_values)):
            ax.annotate(f'K={i+1}', (r, p), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        self._plt.tight_layout()
        
        if save_path:
            self._plt.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot: {save_path}")
        
        return fig
    
    def embedding_visualization(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = "tsne",
        title: str = "Embedding Visualization",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Create 2D visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Optional labels for coloring
            method: Reduction method ("tsne", "umap", "pca")
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size
        """
        self._init_plotting()
        
        # Reduce dimensions
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=config.seed)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, random_state=config.seed)
        else:  # pca
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=config.seed)
        
        coords = reducer.fit_transform(embeddings)
        
        fig, ax = self._plt.subplots(figsize=figsize)
        
        if labels:
            unique_labels = list(set(labels))
            colors = self._sns.color_palette("husl", len(unique_labels))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
            
            for label in unique_labels:
                mask = np.array(labels) == label
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[color_map[label]], label=label, alpha=0.7, s=50)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=50)
        
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        self._plt.tight_layout()
        
        if save_path:
            self._plt.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot: {save_path}")
        
        return fig
    
    def heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Similarity Matrix",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Create heatmap visualization.
        
        Args:
            data: 2D array of values
            row_labels: Row labels
            col_labels: Column labels
            title: Plot title
            save_path: Optional path to save figure
            figsize: Figure size
        """
        self._init_plotting()
        
        fig, ax = self._plt.subplots(figsize=figsize)
        
        im = self._sns.heatmap(
            data,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap='viridis',
            annot=True,
            fmt='.2f',
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        self._plt.xticks(rotation=45, ha='right')
        self._plt.tight_layout()
        
        if save_path:
            self._plt.savefig(self.output_dir / save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot: {save_path}")
        
        return fig


class DashboardGenerator(LoggerMixin):
    """
    Generate HTML dashboards for RAG analysis.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize dashboard generator.
        
        Args:
            output_dir: Directory for saving dashboards
        """
        self.output_dir = Path(output_dir or config.paths.reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        benchmark_results: Dict,
        title: str = "RAG System Benchmark Report"
    ) -> str:
        """
        Generate HTML benchmark report.
        
        Args:
            benchmark_results: Results from benchmarking
            title: Report title
            
        Returns:
            Path to generated HTML file
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-warning {{ color: #f39c12; font-weight: bold; }}
        .metric-bad {{ color: #e74c3c; font-weight: bold; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="timestamp">Generated: {benchmark_results.get('timestamp', 'N/A')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Experiment:</strong> {benchmark_results.get('name', 'N/A')}</p>
            <p><strong>Samples:</strong> {benchmark_results.get('config', {}).get('num_samples', 'N/A')}</p>
            <p><strong>Hallucination Rate:</strong> {benchmark_results.get('hallucination_rate', 0):.1%}</p>
        </div>
        
        <h2>Retrieval Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {self._format_metrics_rows(benchmark_results.get('retrieval_metrics', {}))}
        </table>
        
        <h2>Generation Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {self._format_metrics_rows(benchmark_results.get('generation_metrics', {}))}
        </table>
        
        <h2>Latency Statistics</h2>
        <table>
            <tr>
                <th>Percentile</th>
                <th>Latency (ms)</th>
            </tr>
            {self._format_latency_rows(benchmark_results.get('latency_stats', {}))}
        </table>
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = self.output_dir / f"benchmark_report_{benchmark_results.get('name', 'report')}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Generated report: {report_path}")
        return str(report_path)
    
    def _format_metrics_rows(self, metrics: Dict) -> str:
        """Format metrics as HTML table rows."""
        rows = []
        for name, data in metrics.items():
            value = data.get('value', data) if isinstance(data, dict) else data
            
            # Determine color class
            if isinstance(value, (int, float)):
                if value >= 0.8:
                    css_class = "metric-good"
                elif value >= 0.5:
                    css_class = "metric-warning"
                else:
                    css_class = "metric-bad"
                formatted = f"{value:.4f}"
            else:
                css_class = ""
                formatted = str(value)
            
            rows.append(f'<tr><td>{name}</td><td class="{css_class}">{formatted}</td></tr>')
        
        return "\n".join(rows)
    
    def _format_latency_rows(self, stats: Dict) -> str:
        """Format latency stats as HTML table rows."""
        rows = []
        order = ['mean', 'p50', 'p95', 'p99', 'min', 'max']
        
        for key in order:
            if key in stats:
                value = stats[key]
                css_class = "metric-good" if value < 100 else ("metric-warning" if value < 500 else "metric-bad")
                rows.append(f'<tr><td>{key.upper()}</td><td class="{css_class}">{value:.0f}</td></tr>')
        
        return "\n".join(rows)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization Test")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    if args.test:
        print("Visualization Module Test\n" + "=" * 50)
        
        # Test plot generator
        plotter = PlotGenerator(output_dir=Path("artifacts/plots"))
        
        # Sample metrics
        metrics = {
            "P@5": 0.76,
            "R@5": 0.82,
            "NDCG@5": 0.78,
            "MRR": 0.85,
            "ROUGE-1": 0.65,
            "ROUGE-L": 0.58
        }
        
        print("Sample metrics for visualization:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.2f}")
        
        print("\nDashboard generator initialized")
        dashboard = DashboardGenerator()
        
        # Sample benchmark result
        sample_result = {
            "name": "test_benchmark",
            "timestamp": "2026-02-01T12:00:00",
            "retrieval_metrics": {"ndcg@5": {"value": 0.78}},
            "generation_metrics": {"rouge1": {"value": 0.65}},
            "latency_stats": {"p50": 45, "p95": 120, "p99": 250},
            "hallucination_rate": 0.08,
            "config": {"num_samples": 100}
        }
        
        print(f"Sample benchmark: {sample_result['name']}")
