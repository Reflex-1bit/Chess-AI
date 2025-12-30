"""
Insight Generation from Model Predictions

Aggregates predictions across games to identify recurring weaknesses
and generate human-readable coaching insights.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict


class InsightGenerator:
    """Generates coaching insights from model predictions."""
    
    def __init__(self, mistake_threshold: float = 0.5):
        """
        Initialize insight generator.
        
        Args:
            mistake_threshold: Probability threshold to consider a predicted mistake
        """
        self.mistake_threshold = mistake_threshold
    
    def generate_insights(self, moves_data: List[Dict], predictions: Dict) -> List[str]:
        """
        Generate coaching insights from predictions.
        
        Args:
            moves_data: List of move dictionaries with game context
            predictions: Dictionary with 'mistake_proba' and 'expected_eval_loss' arrays
        
        Returns:
            List of human-readable insight strings
        """
        insights = []
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'game_phase': [m['game_phase'] for m in moves_data],
            'is_white': [m['is_white'] for m in moves_data],
            'move_number': [m['move_number'] for m in moves_data],
            'actual_eval_loss': [m['eval_loss'] for m in moves_data],
            'mistake_proba': predictions['mistake_proba'],
            'expected_eval_loss': predictions['expected_eval_loss']
        })
        
        # 1. Overall mistake rate
        overall_mistake_rate = np.mean(df['mistake_proba'] > self.mistake_threshold)
        avg_eval_loss = df['actual_eval_loss'].mean()
        insights.append(
            f"Overall Performance: {overall_mistake_rate*100:.1f}% of moves are predicted mistakes. "
            f"Average eval loss: {avg_eval_loss:.1f} centipawns."
        )
        
        # 2. Performance by game phase
        phase_insights = self._analyze_by_phase(df)
        insights.extend(phase_insights)
        
        # 3. Performance by material balance
        material_insights = self._analyze_by_material(moves_data, df)
        insights.extend(material_insights)
        
        # 4. Recurring mistake patterns
        pattern_insights = self._find_recurring_patterns(moves_data, df)
        insights.extend(pattern_insights)
        
        # 5. Improvement areas
        improvement_insights = self._identify_improvement_areas(moves_data, df)
        insights.extend(improvement_insights)
        
        return insights
    
    def _analyze_by_phase(self, df: pd.DataFrame) -> List[str]:
        """Analyze performance by game phase."""
        insights = []
        
        for phase in ['opening', 'middlegame', 'endgame']:
            phase_df = df[df['game_phase'] == phase]
            if len(phase_df) == 0:
                continue
            
            mistake_rate = np.mean(phase_df['mistake_proba'] > self.mistake_threshold)
            avg_loss = phase_df['actual_eval_loss'].mean()
            
            if mistake_rate > 0.3 or avg_loss > 100:
                insights.append(
                    f"{phase.capitalize()} Weakness: {mistake_rate*100:.1f}% mistake rate, "
                    f"avg eval loss {avg_loss:.1f} cp. Consider focusing on {phase} study."
                )
        
        return insights
    
    def _analyze_by_material(self, moves_data: List[Dict], df: pd.DataFrame) -> List[str]:
        """Analyze performance when ahead/behind in material."""
        insights = []
        
        # This is simplified - in practice, extract material balance from features
        # For now, use eval_before as proxy (positive = better position)
        material_balance = [m['eval_before'] for m in moves_data]
        
        df['material_balance'] = material_balance
        
        # Ahead (eval > 50 cp)
        ahead_df = df[df['material_balance'] > 50]
        if len(ahead_df) > 10:
            ahead_mistake_rate = np.mean(ahead_df['mistake_proba'] > self.mistake_threshold)
            if ahead_mistake_rate > 0.25:
                insights.append(
                    f"When Ahead: {ahead_mistake_rate*100:.1f}% mistake rate when winning. "
                    "Practice converting winning positions."
                )
        
        # Behind (eval < -50 cp)
        behind_df = df[df['material_balance'] < -50]
        if len(behind_df) > 10:
            behind_mistake_rate = np.mean(behind_df['mistake_proba'] > self.mistake_threshold)
            behind_avg_loss = behind_df['actual_eval_loss'].mean()
            if behind_mistake_rate > 0.35:
                insights.append(
                    f"When Behind: {behind_mistake_rate*100:.1f}% mistake rate when losing, "
                    f"avg loss {behind_avg_loss:.1f} cp. Focus on defensive techniques."
                )
        
        return insights
    
    def _find_recurring_patterns(self, moves_data: List[Dict], df: pd.DataFrame) -> List[str]:
        """Find recurring mistake patterns."""
        insights = []
        
        # Find moves with high mistake probability
        high_risk_moves = df[df['mistake_proba'] > 0.7]
        
        if len(high_risk_moves) > 0:
            # Analyze game phases where mistakes occur
            phase_counts = high_risk_moves['game_phase'].value_counts()
            most_common_phase = phase_counts.index[0] if len(phase_counts) > 0 else None
            
            if most_common_phase:
                insights.append(
                    f"Recurring Pattern: Most mistakes occur in the {most_common_phase}. "
                    f"Consider targeted study in this phase."
                )
        
        return insights
    
    def _identify_improvement_areas(self, moves_data: List[Dict], df: pd.DataFrame) -> List[str]:
        """Identify specific areas for improvement."""
        insights = []
        
        # Compare predicted vs actual mistakes
        predicted_mistakes = df['mistake_proba'] > self.mistake_threshold
        actual_mistakes = df['actual_eval_loss'] > 100  # Using 100 cp threshold
        
        # False positives: Model predicts mistake but it wasn't
        false_positives = predicted_mistakes & ~actual_mistakes
        if false_positives.sum() > len(df) * 0.1:
            insights.append(
                "Model Analysis: You may be playing overly cautious moves. "
                "The model predicts mistakes in positions where you actually play well."
            )
        
        # False negatives: Actual mistakes not predicted
        false_negatives = ~predicted_mistakes & actual_mistakes
        if false_negatives.sum() > len(df) * 0.1:
            insights.append(
                "Model Analysis: Some mistakes occur in unexpected positions. "
                "Review games to identify patterns the model may have missed."
            )
        
        # Consistency check
        mistake_std = df['actual_eval_loss'].std()
        if mistake_std > 200:
            insights.append(
                "Consistency: High variance in move quality. Focus on maintaining consistent play."
            )
        
        return insights
    
    def generate_detailed_report(self, moves_data: List[Dict], predictions: Dict) -> str:
        """
        Generate a detailed coaching report.
        
        Returns:
            Formatted string report
        """
        insights = self.generate_insights(moves_data, predictions)
        
        report = "=" * 60 + "\n"
        report += "CHESS COACHING INSIGHTS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n\n"
        
        report += "=" * 60 + "\n"
        report += "Generated by Human Mistake Modeling System\n"
        report += "=" * 60 + "\n"
        
        return report

