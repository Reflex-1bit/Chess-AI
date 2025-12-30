"""
Lichess Puzzle Themes Integration

Maps blunder types to Lichess puzzle themes for specific coaching advice.
Based on https://lichess.org/training/themes
"""

from typing import Optional, List, Dict

LICHESS_PUZZLE_THEMES = {
    # Phases
    'opening': {
        'name': 'Opening',
        'description': 'A tactic during the first phase of the game.',
        'advice': 'Study opening principles and common tactical patterns that arise in the opening phase.'
    },
    'middlegame': {
        'name': 'Middlegame',
        'description': 'A tactic during the second phase of the game.',
        'advice': 'Focus on middlegame tactics like piece coordination, pawn structure, and tactical combinations.'
    },
    'endgame': {
        'name': 'Endgame',
        'description': 'A tactic during the last phase of the game.',
        'advice': 'Improve your endgame technique. Study basic endgames and tactical motifs specific to endgame positions.'
    },
    
    # Common motifs
    'hanging_piece': {
        'name': 'Hanging Piece',
        'description': 'A tactic involving an opponent piece being undefended or insufficiently defended and free to capture.',
        'advice': 'Always check if your pieces are defended before moving. Look for undefended enemy pieces you can capture.'
    },
    'fork': {
        'name': 'Fork',
        'description': 'A move where the moved piece attacks two opponent pieces at once.',
        'advice': 'Watch for fork opportunities (especially knight forks) and avoid putting your pieces on squares where they can be forked.'
    },
    'pin': {
        'name': 'Pin',
        'description': 'A tactic involving pins, where a piece is unable to move without revealing an attack on a higher value piece.',
        'advice': 'Learn to recognize pins. Avoid moving pinned pieces unless necessary, and look for opportunities to pin enemy pieces.'
    },
    'skewer': {
        'name': 'Skewer',
        'description': 'A motif involving a high value piece being attacked, moving out of the way, and allowing a lower value piece behind it to be captured or attacked.',
        'advice': 'Look for skewer opportunities, especially with rooks and queens. Be careful not to line up your valuable pieces.'
    },
    'discovered_attack': {
        'name': 'Discovered Attack',
        'description': 'Moving a piece that previously blocked an attack by a long range piece out of the way of that piece.',
        'advice': 'Be aware of discovered attacks when moving pieces that block your rooks, bishops, or queen. Check what lines you open.'
    },
    'double_check': {
        'name': 'Double Check',
        'description': 'Checking with two pieces at once, as a result of a discovered attack where both the moving piece and the unveiled piece attack the opponent\'s king.',
        'advice': 'Double checks are powerful - the king must move. Look for discovered check opportunities that also attack the king directly.'
    },
    'sacrifice': {
        'name': 'Sacrifice',
        'description': 'A tactic involving giving up material in the short-term, to gain an advantage again after a forced sequence of moves.',
        'advice': 'Sometimes sacrificing material leads to a winning position. Learn to calculate whether a sacrifice is sound.'
    },
    'trapped_piece': {
        'name': 'Trapped Piece',
        'description': 'A piece is unable to escape capture as it has limited moves.',
        'advice': 'Be careful not to move pieces to squares where they can be trapped. Look for opportunities to trap enemy pieces.'
    },
    'exposed_king': {
        'name': 'Exposed King',
        'description': 'A tactic involving a king with few defenders around it, often leading to checkmate.',
        'advice': 'Keep your king safe. Don\'t expose your king unnecessarily, especially in the middlegame. Look for attacks on exposed enemy kings.'
    },
    'kingside_attack': {
        'name': 'Kingside Attack',
        'description': 'An attack of the opponent\'s king, after they castled on the king side.',
        'advice': 'Learn common kingside attack patterns. When attacking, look for ways to break through the opponent\'s kingside defenses.'
    },
    'queenside_attack': {
        'name': 'Queenside Attack',
        'description': 'An attack of the opponent\'s king, after they castled on the queen side.',
        'advice': 'Queenside attacks require different patterns than kingside attacks. Study both types of attacks.'
    },
    'advanced_pawn': {
        'name': 'Advanced Pawn',
        'description': 'One of your pawns is deep into the opponent position, maybe threatening to promote.',
        'advice': 'Advanced pawns can be powerful but also vulnerable. Support them with pieces and look for promotion opportunities.'
    },
    'attacking_f2_f7': {
        'name': 'Attacking f2/f7',
        'description': 'An attack focusing on the f2 or f7 pawn, such as in the fried liver opening.',
        'advice': 'The f2/f7 squares are weak in the opening. Be aware of attacks on these squares, especially with bishops and knights.'
    },
    'capture_the_defender': {
        'name': 'Capture the Defender',
        'description': 'Removing a piece that is critical to defence of another piece, allowing the now undefended piece to be captured on a following move.',
        'advice': 'Look for pieces that are defending valuable enemy pieces. If you can capture the defender, you may win material.'
    },
    
    # Advanced motifs
    'deflection': {
        'name': 'Deflection',
        'description': 'A move that distracts an opponent piece from another duty that it performs, such as guarding a key square.',
        'advice': 'Deflection tactics distract a piece from an important duty. Look for pieces doing multiple jobs that you can overload.'
    },
    'discovered_check': {
        'name': 'Discovered Check',
        'description': 'Move a piece to reveal a check from a hidden attacking piece, which often leads to a decisive advantage.',
        'advice': 'Discovered checks are very powerful. Look for opportunities to move a piece and reveal a check simultaneously.'
    },
    'interference': {
        'name': 'Interference',
        'description': 'Moving a piece between two opponent pieces to leave one or both opponent pieces undefended.',
        'advice': 'Sometimes you can interfere with enemy piece coordination by placing a piece between them, breaking their connection.'
    },
    'zugzwang': {
        'name': 'Zugzwang',
        'description': 'The opponent is limited in the moves they can make, and all moves worsen their position.',
        'advice': 'In endgames, zugzwang can be decisive. Try to reach positions where your opponent has no good moves.'
    },
    
    # Mate themes
    'back_rank_mate': {
        'name': 'Back Rank Mate',
        'description': 'Checkmate the king on the home rank, when it is trapped there by its own pieces.',
        'advice': 'Watch for back rank weaknesses. Keep an escape square for your king, and look for back rank mate opportunities.'
    },
    'smothered_mate': {
        'name': 'Smothered Mate',
        'description': 'A checkmate delivered by a knight in which the mated king is unable to move because it is surrounded by its own pieces.',
        'advice': 'Smothered mate is a beautiful pattern. Learn to recognize when the enemy king is trapped and vulnerable to a knight checkmate.'
    },
}


def get_theme_from_blunder_type(blunder_type: str) -> Optional[dict]:
    """
    Map a blunder type to a Lichess puzzle theme.
    
    Args:
        blunder_type: Type of blunder (e.g., 'hanging_piece', 'fork', 'pin')
        
    Returns:
        Theme dictionary with name, description, and advice, or None
    """
    # Normalize the blunder type
    blunder_type_lower = blunder_type.lower().replace(' ', '_').replace('-', '_')
    
    # Direct mapping
    if blunder_type_lower in LICHESS_PUZZLE_THEMES:
        return LICHESS_PUZZLE_THEMES[blunder_type_lower]
    
    # Partial matches
    for theme_key, theme_data in LICHESS_PUZZLE_THEMES.items():
        if theme_key in blunder_type_lower or blunder_type_lower in theme_key:
            return theme_data
    
    return None


def get_themes_for_weakness(weakness_description: str) -> List[dict]:
    """
    Suggest puzzle themes based on a weakness description.
    
    Args:
        weakness_description: Text description of a weakness (e.g., "endgame", "tactics", "king safety")
        
    Returns:
        List of relevant theme dictionaries
    """
    weakness_lower = weakness_description.lower()
    suggested_themes = []
    
    # Map keywords to themes
    keyword_mapping = {
        'endgame': ['endgame', 'rook_endgame', 'pawn_endgame', 'zugzwang'],
        'opening': ['opening'],
        'middlegame': ['middlegame'],
        'tactic': ['fork', 'pin', 'skewer', 'discovered_attack', 'double_check'],
        'king': ['exposed_king', 'kingside_attack', 'queenside_attack', 'back_rank_mate'],
        'piece': ['hanging_piece', 'trapped_piece'],
        'material': ['fork', 'pin', 'skewer', 'sacrifice'],
        'pawn': ['advanced_pawn', 'pawn_endgame'],
    }
    
    for keyword, theme_keys in keyword_mapping.items():
        if keyword in weakness_lower:
            for theme_key in theme_keys:
                if theme_key in LICHESS_PUZZLE_THEMES:
                    suggested_themes.append(LICHESS_PUZZLE_THEMES[theme_key])
    
    return suggested_themes[:3]  # Return top 3 most relevant

