"""
Opening Database and Line Explanations

Provides opening identification and explanations for common openings.
"""

import chess
from typing import Optional, Dict, List, Tuple


class OpeningDatabase:
    """Database of chess openings with explanations."""
    
    def __init__(self):
        """Initialize opening database."""
        self.openings = self._load_openings()
    
    def _load_openings(self) -> Dict[str, Dict]:
        """Load opening database."""
        return {
            # King's Pawn Openings
            "e4 e5": {
                "name": "King's Pawn Game",
                "variation": "Open Game",
                "explanation": "The most common opening. Both players develop their kingside pieces and fight for the center."
            },
            "e4 e5 Nf3 Nc6": {
                "name": "Italian Game",
                "variation": "Classical",
                "explanation": "White develops the bishop to c4, aiming for quick development and control of the center. The Italian Game is aggressive and tactical."
            },
            "e4 e5 Nf3 Nc6 Bb5": {
                "name": "Ruy Lopez",
                "variation": "Spanish Opening",
                "explanation": "One of the oldest and most respected openings. White pins the knight and prepares to castle, with long-term pressure on Black's position."
            },
            "e4 e5 Nf3 Nc6 Bc4": {
                "name": "Italian Game",
                "variation": "Giuoco Piano",
                "explanation": "White develops the bishop to c4, putting immediate pressure on f7. This leads to open, tactical positions."
            },
            "e4 e5 Nf3 Nf6": {
                "name": "Petrov Defense",
                "variation": "Russian Defense",
                "explanation": "Black mirrors White's moves, leading to symmetrical positions. Solid but can be drawish."
            },
            "e4 c5": {
                "name": "Sicilian Defense",
                "variation": "Open Sicilian",
                "explanation": "The most popular response to e4. Black fights for the center asymmetrically, leading to complex, tactical positions."
            },
            "e4 c6": {
                "name": "Caro-Kann Defense",
                "variation": "Classical",
                "explanation": "A solid, defensive opening. Black prepares d5 to challenge White's center, leading to closed positions."
            },
            "e4 e6": {
                "name": "French Defense",
                "variation": "Classical",
                "explanation": "Black builds a solid pawn structure but can have trouble with the light-squared bishop. Leads to strategic battles."
            },
            "e4 d6": {
                "name": "Pirc Defense",
                "variation": "Modern Defense",
                "explanation": "Black allows White to build a strong center, then attacks it later. Hypermodern approach."
            },
            
            # Queen's Pawn Openings
            "d4 d5": {
                "name": "Queen's Gambit Declined",
                "variation": "Orthodox Defense",
                "explanation": "Black challenges White's center. Black declines the gambit pawn, leading to strategic, positional games."
            },
            "d4 d5 c4": {
                "name": "Queen's Gambit",
                "variation": "Accepted",
                "explanation": "White offers a pawn to gain central control. If Black accepts, White gets rapid development."
            },
            "d4 Nf6": {
                "name": "Indian Defense",
                "variation": "King's Indian",
                "explanation": "Black develops the knight first, preparing to fianchetto the bishop. Leads to complex, double-edged positions."
            },
            "d4 Nf6 c4 g6": {
                "name": "King's Indian Defense",
                "variation": "Classical",
                "explanation": "Black allows White to build a strong center, then attacks it with pawn breaks. Very aggressive."
            },
            "d4 Nf6 c4 e6": {
                "name": "Nimzo-Indian Defense",
                "variation": "Classical",
                "explanation": "Black develops the bishop to b4, pinning the knight. Leads to rich, strategic positions."
            },
            "d4 Nf6 c4 c5": {
                "name": "Benoni Defense",
                "variation": "Modern Benoni",
                "explanation": "Black creates an asymmetrical pawn structure, leading to sharp, tactical positions."
            },
            
            # Other Openings
            "Nf3": {
                "name": "Reti Opening",
                "variation": "Zukertort Opening",
                "explanation": "A flexible opening that can transpose into many different systems. White keeps options open."
            },
            "c4": {
                "name": "English Opening",
                "variation": "Reversed Sicilian",
                "explanation": "White plays on the flank, controlling d5. Can transpose into many different openings."
            },
            "f4": {
                "name": "Bird's Opening",
                "variation": "From's Gambit",
                "explanation": "An aggressive flank opening. White aims for a quick kingside attack but weakens the king's position."
            },
            "b3": {
                "name": "Nimzo-Larsen Attack",
                "variation": "Larsen's Opening",
                "explanation": "A hypermodern opening. White fianchettos the bishop and plays for long-term pressure."
            }
        }
    
    def identify_opening(self, moves: List[str], max_moves: int = 10) -> Optional[Dict]:
        """
        Identify the opening from a sequence of moves.
        
        Args:
            moves: List of moves in SAN notation
            max_moves: Maximum number of moves to check
        
        Returns:
            Opening dictionary with name, variation, and explanation, or None
        """
        if not moves:
            return None
        
        # Try progressively longer move sequences
        for length in range(min(max_moves, len(moves)), 0, -1):
            move_sequence = " ".join(moves[:length])
            
            # Check for exact match
            if move_sequence in self.openings:
                return self.openings[move_sequence]
            
            # Check for partial matches (longer sequences first)
            sorted_keys = sorted(self.openings.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if move_sequence.startswith(key):
                    return self.openings[key]
        
        return None
    
    def get_opening_from_game(self, moves_data: List[Dict], max_moves: int = 10) -> Optional[Dict]:
        """
        Identify opening from game moves data.
        
        Args:
            moves_data: List of move dictionaries
            max_moves: Maximum number of moves to check
        
        Returns:
            Opening dictionary or None
        """
        moves = [m.get('move_san', '') for m in moves_data[:max_moves]]
        moves = [m for m in moves if m]  # Filter empty moves
        
        return self.identify_opening(moves, max_moves)
    
    def get_opening_explanation(self, opening_name: str) -> Optional[str]:
        """
        Get explanation for a specific opening.
        
        Args:
            opening_name: Name of the opening
        
        Returns:
            Explanation string or None
        """
        for opening_data in self.openings.values():
            if opening_data['name'] == opening_name:
                return opening_data['explanation']
        return None
    
    def get_all_openings(self) -> List[Dict]:
        """Get all openings in the database."""
        return list(self.openings.values())

