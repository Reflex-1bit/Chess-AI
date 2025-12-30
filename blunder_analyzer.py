"""
Blunder Analysis and Detailed Feedback

Provides specific blunder identification and explanations.
"""

import chess
import chess.engine
from typing import List, Dict, Optional, Tuple
import numpy as np
try:
    from puzzle_themes import get_theme_from_blunder_type
except ImportError:
    get_theme_from_blunder_type = None


class BlunderAnalyzer:
    """Analyzes blunders and provides detailed feedback."""
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """Initialize blunder analyzer."""
        self.stockfish_path = stockfish_path
        self.engine = None
    
    def _get_engine(self):
        """Get Stockfish engine for analysis."""
        if self.engine is None and self.stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception:
                pass
        return self.engine
    
    def identify_blunders(self, moves_data: List[Dict], 
                          blunder_threshold: float = 200.0) -> List[Dict]:
        """
        Identify all blunders in the game.
        
        Args:
            moves_data: List of move dictionaries
            blunder_threshold: Eval loss threshold for blunder (centipawns)
        
        Returns:
            List of blunder dictionaries with detailed analysis
        """
        blunders = []
        
        for i, move_data in enumerate(moves_data):
            eval_loss = move_data.get('eval_loss', 0)
            
            if eval_loss > blunder_threshold:
                blunder_info = {
                    'move_index': i,
                    'move_number': move_data.get('move_number', i + 1),
                    'move_san': move_data.get('move_san', ''),
                    'eval_loss': eval_loss,
                    'eval_before': move_data.get('eval_before', 0),
                    'eval_after': move_data.get('eval_after', 0),
                    'board_before': move_data.get('board_before'),
                    'board_after': move_data.get('board_after'),
                    'is_white': move_data.get('is_white', True),
                    'game_phase': move_data.get('game_phase', 'unknown'),
                    'explanation': self._explain_blunder(move_data, eval_loss)
                }
                blunders.append(blunder_info)
        
        return blunders
    
    def format_eval(self, cp: float) -> str:
        """Format evaluation in a user-friendly way with Stockfish evaluation terms."""
        if abs(cp) >= 10000:
            return "Checkmate"
        pawns = cp / 100.0
        if pawns >= 0:
            return f"+{pawns:.1f}"
        else:
            return f"{pawns:.1f}"
    
    def format_eval_loss(self, cp: float) -> str:
        """Format evaluation loss as blunder points (using Stockfish evaluation)."""
        if abs(cp) >= 10000:
            return "checkmate"
        
        # Convert centipawns to blunder points (1 point = 1 pawn of evaluation)
        blunder_points = abs(cp) / 100.0
        
        # Round to 1 decimal place, but show as integer if it's a whole number
        if blunder_points >= 1.0:
            points = int(blunder_points) if blunder_points == int(blunder_points) else round(blunder_points, 1)
            return f"{points} blunder point{'s' if points != 1 else ''}"
        elif blunder_points >= 0.5:
            return f"{round(blunder_points, 1)} blunder points"
        else:
            # For very small losses, use more precise language
            return f"{round(blunder_points, 2)} blunder points"
    
    def _explain_blunder(self, move_data: Dict, eval_loss: float) -> str:
        """
        Generate explanation for why a move is a blunder.
        
        Args:
            move_data: Move dictionary
            eval_loss: Evaluation loss in centipawns
        
        Returns:
            Explanation string
        """
        explanations = []
        
        # Severity - use blunder points
        blunder_points = eval_loss / 100.0
        if eval_loss > 500:
            explanations.append(f"**Critical Blunder**: Lost {blunder_points:.1f} blunder points (Stockfish evaluation)")
        elif eval_loss > 300:
            explanations.append(f"**Major Blunder**: Lost {blunder_points:.1f} blunder points (Stockfish evaluation)")
        else:
            explanations.append(f"**Blunder**: Lost {blunder_points:.1f} blunder points (Stockfish evaluation)")
        
        # Position context
        eval_before = move_data.get('eval_before', 0)
        eval_after = move_data.get('eval_after', 0)
        
        if eval_before > 100:
            explanations.append("You were in a **winning position** before this move.")
        elif eval_before > 0:
            explanations.append("You had a **slight advantage** before this move.")
        elif eval_before < -100:
            explanations.append("You were already in a **difficult position**.")
        
        if eval_after < -200:
            explanations.append("After this move, you're in a **losing position**.")
        elif eval_after < -100:
            explanations.append("After this move, you're at a **significant disadvantage**.")
        
        # Game phase context
        game_phase = move_data.get('game_phase', 'unknown')
        if game_phase == 'opening':
            explanations.append("This blunder occurred in the **opening phase**. Consider studying opening theory.")
        elif game_phase == 'endgame':
            explanations.append("This blunder occurred in the **endgame**. Endgame technique needs improvement.")
        
        # Try to identify the type of blunder
        board_before = move_data.get('board_before')
        board_after = move_data.get('board_after')
        
        if board_before and board_after:
            blunder_type_text, blunder_type_key = self._identify_blunder_type(board_before, board_after, move_data)
            if blunder_type_text:
                explanations.append(blunder_type_text)
            
            # Get theme-specific advice from Lichess themes
            if blunder_type_key and get_theme_from_blunder_type:
                theme = get_theme_from_blunder_type(blunder_type_key)
                if theme:
                    explanations.append(f"**Study Theme:** {theme['name']} - {theme['advice']}")
        
        return " ".join(explanations)
    
    def _identify_blunder_type(self, board_before: chess.Board, 
                              board_after: chess.Board,
                              move_data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify the type of blunder and what was missed.
        
        Returns:
            Tuple of (explanation_text, theme_key) where theme_key can be used for puzzle theme lookup
        """
        explanations = []
        theme_key = None
        move_san = move_data.get('move_san', '')
        is_white = move_data.get('is_white', True)
        
        # Get the move that was played
        played_move = None
        for move in board_before.legal_moves:
            test_board = board_before.copy()
            test_board.push(move)
            if test_board.fen() == board_after.fen():
                played_move = move
                break
        
        # Check for hanging pieces (lost material)
        pieces_before = len(board_before.piece_map())
        pieces_after = len(board_after.piece_map())
        
        if pieces_after < pieces_before:
            # Identify what piece was lost
            for square in range(64):
                piece_before = board_before.piece_at(square)
                piece_after = board_after.piece_at(square)
                if piece_before and (not piece_after or piece_after.color != piece_before.color):
                    if piece_before.color == (chess.WHITE if is_white else chess.BLACK):
                        piece_name = piece_before.symbol().upper() if is_white else piece_before.symbol()
                        explanations.append(f"**Material Lost**: You hung a {piece_name}. Always check if your pieces are defended before moving.")
                        theme_key = 'hanging_piece'
                        break
        
        # Check if opponent can capture something now
        opponent_color = chess.BLACK if is_white else chess.WHITE
        for square in range(64):
            piece = board_after.piece_at(square)
            if piece and piece.color == (chess.WHITE if is_white else chess.BLACK):
                attackers = board_after.attackers(opponent_color, square)
                defenders = board_after.attackers(chess.WHITE if is_white else chess.BLACK, square)
                if len(attackers) > len(defenders):
                    explanations.append(f"**Hanging Piece**: Your piece on {chess.square_name(square)} is now undefended and can be captured.")
                    if not theme_key:
                        theme_key = 'hanging_piece'
        
        # Check for check and tactical threats
        if board_after.is_check():
            explanations.append("**Tactical Mistake**: You gave check, but this likely created a tactical opportunity for your opponent.")
        
        # Check for discovered attacks
        if played_move:
            # Check if moving this piece opened up an attack
            from_square = played_move.from_square
            to_square = played_move.to_square
            piece_moved = board_before.piece_at(from_square)
            
            if piece_moved:
                # Check if there's a piece behind the moved piece that can now attack
                # This is simplified - check if opponent pieces can now attack our king
                our_king = board_after.king(chess.WHITE if is_white else chess.BLACK)
                if our_king is not None:
                    attackers = board_after.attackers(opponent_color, our_king)
                    if len(attackers) > 0:
                        explanations.append("**King Safety**: Moving this piece exposed your king to attack. Always consider what lines you're opening.")
        
        # Check for missed tactical opportunities
        # Look for forks, pins, skewers that the opponent can now play
        opponent_moves = list(board_after.legal_moves)
        for move in opponent_moves[:10]:  # Check first 10 moves
            test_board = board_after.copy()
            test_board.push(move)
            
            # Check for forks
            for square in range(64):
                piece = test_board.piece_at(square)
                if piece and piece.color == (chess.WHITE if is_white else chess.BLACK):
                    attackers = test_board.attackers(opponent_color, square)
                    if len(attackers) >= 2:
                        explanations.append("**Tactical Threat**: Your move allows a fork or double attack. Always check for tactical threats before moving.")
                        if not theme_key:
                            theme_key = 'fork'
                        break
            if explanations:
                break
        
        # Check for pawn structure weaknesses
        if played_move and board_before.piece_at(played_move.from_square):
            moved_piece = board_before.piece_at(played_move.from_square)
            if moved_piece and moved_piece.piece_type == chess.PAWN:
                # Check for isolated or doubled pawns
                file = chess.square_file(played_move.to_square)
                explanations.append("**Pawn Structure**: This pawn move may have weakened your pawn structure. Consider pawn structure before advancing.")
        
        # If no specific issue found, provide general advice
        if not explanations:
            explanations.append("**Positional Error**: Your move weakened the position. Consider piece coordination, king safety, and control of key squares.")
        
        explanation_text = " ".join(explanations) if explanations else None
        return (explanation_text, theme_key)
    
    def get_best_alternative(self, board: chess.Board, 
                           depth: int = 15) -> Optional[Tuple[str, float]]:
        """
        Get the best alternative move for a position.
        
        Args:
            board: Chess board position
            depth: Analysis depth
        
        Returns:
            Tuple of (move_san, evaluation) or None
        """
        engine = self._get_engine()
        if not engine:
            return None
        
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = info.get("pv", [None])[0] if info.get("pv") else None
            
            if best_move:
                board_copy = board.copy()
                board_copy.push(best_move)
                eval_info = engine.analyse(board_copy, chess.engine.Limit(depth=depth))
                score = eval_info.get("score")
                
                if score:
                    if hasattr(score, 'relative'):
                        score = score.relative
                    if score.is_mate():
                        eval_cp = 10000 if score.mate() > 0 else -10000
                    else:
                        eval_cp = score.score()
                    
                    move_san = board.san(best_move)
                    return (move_san, eval_cp)
        except Exception:
            pass
        
        return None
    
    def close(self):
        """Close the engine."""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None

