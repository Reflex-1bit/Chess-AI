"""
AI-Powered Puzzle Recommendation System

Analyzes coaching feedback and user weaknesses to recommend personalized puzzles.
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import chess
from lichess_puzzles import LichessPuzzleAPI


class PuzzleRecommender:
    """Recommends puzzles based on AI coaching insights and user weaknesses."""
    
    def __init__(self):
        """Initialize puzzle recommender."""
        self.puzzle_api = LichessPuzzleAPI()
        
        # Map coaching insights to Lichess puzzle themes
        self.theme_mapping = {
            'opening': ['opening', 'advancedPawn'],
            'middlegame': ['middlegame', 'attackingF2F7', 'crushing'],
            'endgame': ['endgame', 'advancedPawn'],
            'tactical': ['fork', 'pin', 'skewer', 'discoveredAttack', 'doubleCheck'],
            'sacrifice': ['sacrifice', 'crushing'],
            'mate': ['mateIn1', 'mateIn2', 'mateIn3'],
            'material': ['hangingPiece', 'trappedPiece', 'capturingDefender'],
            'positional': ['zugzwang', 'advancedPawn'],
            'kingside_attack': ['kingsideAttack', 'attackingF2F7'],
            'queenside_attack': ['queensideAttack']
        }
        
        # Theme priority weights (can be learned from user performance)
        self.theme_weights = defaultdict(lambda: 1.0)
    
    def analyze_weaknesses(self, moves_data: List[Dict], 
                          predictions: Optional[Dict] = None,
                          insights: Optional[List[str]] = None) -> Dict:
        """
        Analyze user weaknesses from coaching data.
        
        Args:
            moves_data: List of move dictionaries
            predictions: Model predictions (optional)
            insights: List of insight strings (optional)
        
        Returns:
            Dictionary of weakness analysis with theme priorities
        """
        weaknesses = {
            'game_phases': defaultdict(int),
            'tactical_patterns': defaultdict(int),
            'eval_losses': [],
            'blunder_patterns': []
        }
        
        # Analyze by game phase
        for move in moves_data:
            phase = move.get('game_phase', 'unknown')
            eval_loss = move.get('eval_loss', 0)
            
            if eval_loss > 100:  # Significant mistakes
                weaknesses['game_phases'][phase] += 1
                weaknesses['eval_losses'].append(eval_loss)
        
        # Extract themes from insights if available
        if insights:
            for insight in insights:
                insight_lower = insight.lower()
                
                # Check for phase mentions
                if 'opening' in insight_lower:
                    weaknesses['tactical_patterns']['opening'] += 2
                if 'middlegame' in insight_lower or 'middle game' in insight_lower:
                    weaknesses['tactical_patterns']['middlegame'] += 2
                if 'endgame' in insight_lower or 'end game' in insight_lower:
                    weaknesses['tactical_patterns']['endgame'] += 2
                
                # Check for tactical patterns
                if 'fork' in insight_lower or 'double attack' in insight_lower:
                    weaknesses['tactical_patterns']['fork'] += 3
                if 'pin' in insight_lower:
                    weaknesses['tactical_patterns']['pin'] += 3
                if 'skewer' in insight_lower:
                    weaknesses['tactical_patterns']['skewer'] += 3
                if 'discovered' in insight_lower:
                    weaknesses['tactical_patterns']['discoveredAttack'] += 3
                if 'sacrifice' in insight_lower:
                    weaknesses['tactical_patterns']['sacrifice'] += 3
                if 'hanging' in insight_lower or 'undefended' in insight_lower:
                    weaknesses['tactical_patterns']['material'] += 2
                if 'mate' in insight_lower or 'checkmate' in insight_lower:
                    weaknesses['tactical_patterns']['mate'] += 3
                if 'king safety' in insight_lower or 'king exposed' in insight_lower:
                    weaknesses['tactical_patterns']['kingside_attack'] += 2
        
        # Calculate priority scores for themes
        theme_priorities = defaultdict(float)
        
        # Phase-based priorities
        total_phase_mistakes = sum(weaknesses['game_phases'].values())
        if total_phase_mistakes > 0:
            for phase, count in weaknesses['game_phases'].items():
                priority = count / total_phase_mistakes
                for theme in self.theme_mapping.get(phase, []):
                    theme_priorities[theme] += priority * 2
        
        # Tactical pattern priorities
        total_tactical = sum(weaknesses['tactical_patterns'].values())
        if total_tactical > 0:
            for pattern, count in weaknesses['tactical_patterns'].items():
                priority = count / total_tactical
                for theme in self.theme_mapping.get(pattern, [pattern]):
                    theme_priorities[theme] += priority * 3
        
        # Normalize priorities
        max_priority = max(theme_priorities.values()) if theme_priorities else 1.0
        if max_priority > 0:
            theme_priorities = {k: v / max_priority for k, v in theme_priorities.items()}
        
        return {
            'weaknesses': weaknesses,
            'recommended_themes': dict(sorted(theme_priorities.items(), key=lambda x: x[1], reverse=True)),
            'primary_weakness': max(weaknesses['game_phases'].items(), key=lambda x: x[1])[0] if weaknesses['game_phases'] else None
        }
    
    def get_recommended_themes(self, analysis: Dict, top_n: int = 5) -> List[str]:
        """
        Get top recommended puzzle themes.
        
        Args:
            analysis: Weakness analysis dictionary
            top_n: Number of themes to return
        
        Returns:
            List of recommended theme names
        """
        recommended = analysis.get('recommended_themes', {})
        sorted_themes = sorted(recommended.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes[:top_n] if score > 0]
    
    def get_puzzle_recommendations(self, 
                                   moves_data: Optional[List[Dict]] = None,
                                   predictions: Optional[Dict] = None,
                                   insights: Optional[List[str]] = None,
                                   user_rating: int = 1500) -> Dict:
        """
        Get puzzle recommendations based on coaching analysis.
        
        Args:
            moves_data: User's game moves
            predictions: Model predictions
            insights: Coaching insights
            user_rating: User's approximate rating
        
        Returns:
            Dictionary with recommendations and theme priorities
        """
        # Analyze weaknesses
        if moves_data:
            analysis = self.analyze_weaknesses(moves_data, predictions, insights)
        else:
            # Default analysis if no data available
            analysis = {
                'weaknesses': {'game_phases': {}, 'tactical_patterns': {}},
                'recommended_themes': {},
                'primary_weakness': None
            }
        
        # Get recommended themes
        recommended_themes = self.get_recommended_themes(analysis, top_n=5)
        
        # Calculate rating range (user rating ± 200)
        rating_min = max(1000, user_rating - 200)
        rating_max = min(2500, user_rating + 200)
        
        return {
            'analysis': analysis,
            'recommended_themes': recommended_themes,
            'rating_range': (rating_min, rating_max),
            'recommendation_text': self._generate_recommendation_text(analysis, recommended_themes)
        }
    
    def _generate_recommendation_text(self, analysis: Dict, themes: List[str]) -> str:
        """Generate human-readable recommendation text."""
        if not themes:
            return "Start with general tactical puzzles to improve your overall skills."
        
        primary = themes[0] if themes else None
        weakness = analysis.get('primary_weakness')
        
        recommendations = []
        
        if primary:
            theme_descriptions = {
                'opening': 'opening principles and development',
                'middlegame': 'middlegame tactics and strategy',
                'endgame': 'endgame technique',
                'fork': 'forks and double attacks',
                'pin': 'pins and immobilization',
                'skewer': 'skewers and line attacks',
                'discoveredAttack': 'discovered attacks',
                'sacrifice': 'sacrificial combinations',
                'mateIn1': 'mate in 1 patterns',
                'mateIn2': 'mate in 2 combinations',
                'mateIn3': 'mate in 3 sequences',
                'hangingPiece': 'hanging pieces and undefended material',
                'attackingF2F7': 'attacking weak squares (f2/f7)',
                'kingsideAttack': 'kingside attacks'
            }
            
            desc = theme_descriptions.get(primary, primary)
            recommendations.append(f"**Focus on {desc}** - This is your primary weakness area.")
        
        if weakness:
            recommendations.append(f"Your analysis shows you struggle most in the **{weakness} phase**.")
        
        if len(themes) > 1:
            recommendations.append(f"Also practice: {', '.join(themes[1:4])}")
        
        return " ".join(recommendations) if recommendations else "Practice a variety of tactical puzzles."

