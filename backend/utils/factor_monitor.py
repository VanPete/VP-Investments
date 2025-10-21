"""
Factor-Level Monitoring System
================================

Track factor calculation success rates, identify problematic factors,
and provide insights for data quality improvements.

Features:
- Per-factor success/failure tracking
- Error pattern analysis
- Coverage reporting
- Automated alerts for low-performing factors
"""

from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorStats:
    """Statistics for a single factor across multiple tickers"""
    factor_name: str
    success_count: int = 0
    failure_count: int = 0
    errors: List[str] = field(default_factory=list)
    null_values: int = 0
    invalid_values: int = 0  # inf, nan after calculation
    
    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts
    
    @property
    def coverage_rate(self) -> float:
        """Success rate excluding null/invalid values"""
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts


@dataclass
class GroupStats:
    """Aggregated statistics for a factor group"""
    group_name: str
    total_factors: int = 0
    avg_success_rate: float = 0.0
    avg_coverage_rate: float = 0.0
    problematic_factors: List[str] = field(default_factory=list)


class FactorMonitor:
    """
    Monitor factor calculation success rates and identify issues.
    
    Usage:
        monitor = FactorMonitor()
        
        # During calculation loop
        for ticker in tickers:
            for factor_name, value in calculated_factors.items():
                if value is not None:
                    monitor.record_success(factor_name)
                else:
                    monitor.record_failure(factor_name, "calculation_failed")
        
        # At end of batch
        monitor.report(min_success_rate=0.7)
        monitor.save_report('logs/factor_monitoring_20251017.json')
    """
    
    def __init__(self):
        self.stats: Dict[str, FactorStats] = defaultdict(
            lambda: FactorStats(factor_name="")
        )
        self.group_mapping: Dict[str, str] = {}  # factor -> group
        self.start_time = datetime.now()
        
    def set_group_mapping(self, mapping: Dict[str, str]):
        """Set factor-to-group mapping for group-level reporting"""
        self.group_mapping = mapping
    
    def record_success(self, factor_name: str):
        """Record successful factor calculation"""
        self.stats[factor_name].factor_name = factor_name
        self.stats[factor_name].success_count += 1
    
    def record_failure(self, factor_name: str, error: str):
        """Record failed factor calculation"""
        self.stats[factor_name].factor_name = factor_name
        self.stats[factor_name].failure_count += 1
        self.stats[factor_name].errors.append(error)
    
    def record_null(self, factor_name: str):
        """Record null value (missing data)"""
        self.stats[factor_name].factor_name = factor_name
        self.stats[factor_name].null_values += 1
    
    def record_invalid(self, factor_name: str):
        """Record invalid value (inf, nan)"""
        self.stats[factor_name].factor_name = factor_name
        self.stats[factor_name].invalid_values += 1
    
    def get_factor_stats(self, factor_name: str) -> Optional[FactorStats]:
        """Get statistics for a specific factor"""
        return self.stats.get(factor_name)
    
    def get_group_stats(self, group_name: str) -> GroupStats:
        """Get aggregated statistics for a factor group"""
        group_factors = [
            fname for fname, grp in self.group_mapping.items()
            if grp == group_name
        ]
        
        if not group_factors:
            return GroupStats(group_name=group_name)
        
        group_stats = [self.stats[f] for f in group_factors if f in self.stats]
        
        if not group_stats:
            return GroupStats(group_name=group_name)
        
        avg_success = sum(s.success_rate for s in group_stats) / len(group_stats)
        avg_coverage = sum(s.coverage_rate for s in group_stats) / len(group_stats)
        
        # Find problematic factors (success rate < 50%)
        problematic = [
            s.factor_name for s in group_stats
            if s.success_rate < 0.5 and s.total_attempts > 0
        ]
        
        return GroupStats(
            group_name=group_name,
            total_factors=len(group_factors),
            avg_success_rate=avg_success,
            avg_coverage_rate=avg_coverage,
            problematic_factors=problematic
        )
    
    def report(self, min_success_rate: float = 0.7) -> Dict[str, any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            min_success_rate: Threshold for alerting (default 70%)
            
        Returns:
            Dict with summary statistics and alerts
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("FACTOR MONITORING REPORT")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Total factors tracked: {len(self.stats)}")
        
        # Overall statistics
        total_attempts = sum(s.total_attempts for s in self.stats.values())
        total_successes = sum(s.success_count for s in self.stats.values())
        overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0
        
        logger.info(f"Overall success rate: {overall_success_rate:.1%}")
        logger.info(f"Total calculations: {total_attempts:,}")
        logger.info(f"Successful: {total_successes:,}")
        logger.info(f"Failed: {total_attempts - total_successes:,}")
        
        # Find problematic factors
        logger.info("\n" + "-" * 80)
        logger.info("LOW SUCCESS RATE FACTORS (< {:.0%})".format(min_success_rate))
        logger.info("-" * 80)
        
        problematic_factors = []
        for factor_name, stats in sorted(self.stats.items()):
            if stats.total_attempts == 0:
                continue
            
            if stats.success_rate < min_success_rate:
                problematic_factors.append((factor_name, stats))
                
                group = self.group_mapping.get(factor_name, "unknown")
                logger.warning(
                    f"[{group:>25}] {factor_name:40} "
                    f"{stats.success_rate:6.1%} success "
                    f"({stats.success_count}/{stats.total_attempts}) ⚠️"
                )
                
                # Show most common errors
                if stats.errors:
                    error_counts = Counter(stats.errors)
                    for error, count in error_counts.most_common(3):
                        logger.warning(f"  → {error}: {count} occurrences")
        
        # Group-level summary
        logger.info("\n" + "-" * 80)
        logger.info("GROUP-LEVEL SUMMARY")
        logger.info("-" * 80)
        
        group_stats = {}
        for group_name in ['technical', 'fundamental', 'news_macro', 
                          'social_alternative', 'risk_stability', 
                          'institutional_smart_money']:
            gstats = self.get_group_stats(group_name)
            group_stats[group_name] = gstats
            
            status = "✅" if gstats.avg_success_rate >= 0.7 else "⚠️" if gstats.avg_success_rate >= 0.5 else "❌"
            
            logger.info(
                f"{status} {group_name:30} "
                f"{gstats.avg_success_rate:6.1%} avg success "
                f"({gstats.total_factors} factors, "
                f"{len(gstats.problematic_factors)} problematic)"
            )
        
        logger.info("=" * 80)
        
        # Return structured report
        return {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'overall_success_rate': overall_success_rate,
            'total_factors': len(self.stats),
            'total_calculations': total_attempts,
            'problematic_factors': [
                {
                    'factor': name,
                    'success_rate': stats.success_rate,
                    'attempts': stats.total_attempts,
                    'top_errors': dict(Counter(stats.errors).most_common(3))
                }
                for name, stats in problematic_factors
            ],
            'group_summary': {
                name: {
                    'avg_success_rate': gstats.avg_success_rate,
                    'total_factors': gstats.total_factors,
                    'problematic_count': len(gstats.problematic_factors),
                    'problematic_factors': gstats.problematic_factors
                }
                for name, gstats in group_stats.items()
            }
        }
    
    def save_report(self, filepath: str):
        """Save monitoring report to JSON file"""
        import json
        from pathlib import Path
        
        report = self.report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Monitoring report saved: {filepath}")
    
    def get_recommendations(self) -> List[Tuple[str, str, str]]:
        """
        Generate improvement recommendations based on monitoring data.
        
        Returns:
            List of (factor_name, issue, recommendation) tuples
        """
        recommendations = []
        
        for factor_name, stats in self.stats.items():
            if stats.total_attempts == 0:
                continue
            
            group = self.group_mapping.get(factor_name, "unknown")
            
            # Low success rate
            if stats.success_rate < 0.5:
                if stats.errors:
                    top_error = Counter(stats.errors).most_common(1)[0][0]
                    
                    if "KeyError" in top_error or "AttributeError" in top_error:
                        recommendations.append((
                            factor_name,
                            f"Missing data source ({stats.success_rate:.0%} success)",
                            f"Add fallback data source or remove factor from {group} group"
                        ))
                    elif "ZeroDivisionError" in top_error:
                        recommendations.append((
                            factor_name,
                            f"Division by zero ({stats.success_rate:.0%} success)",
                            "Add denominator validation or use safe division"
                        ))
                    else:
                        recommendations.append((
                            factor_name,
                            f"Calculation errors ({stats.success_rate:.0%} success)",
                            f"Review calculation logic in {group} group"
                        ))
        
        return recommendations
