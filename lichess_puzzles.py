"""
Lichess Puzzles API Integration

Fetches puzzles from Lichess and provides interactive puzzle solving interface.
"""

import requests
import chess
import chess.engine
from typing import Optional, Dict, List, Tuple
import streamlit as st


class LichessPuzzleAPI:
    """Interface for fetching puzzles from Lichess API."""
    
    BASE_URL = "https://lichess.org/api"
    
    def __init__(self):
        """Initialize Lichess puzzle API client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChessCoachingSystem/1.0'
        })
    
    def get_daily_puzzle(self) -> Optional[Dict]:
        """
        Fetch the daily puzzle from Lichess.
        
        Returns:
            Puzzle dictionary with 'puzzle', 'game', 'puzzle' fields or None
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/puzzle/daily", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching daily puzzle: {e}")
            return None
    
    def get_puzzle_by_rating(self, rating_min: int = 1500, rating_max: int = 2000) -> Optional[Dict]:
        """
        Fetch a random puzzle within a rating range.
        
        Note: Lichess doesn't have a direct endpoint for this, but we can use
        the puzzle activity endpoint or fetch from puzzle database.
        For now, we'll use the daily puzzle as a fallback.
        
        Args:
            rating_min: Minimum puzzle rating
            rating_max: Maximum puzzle rating
            
        Returns:
            Puzzle dictionary or None
        """
        # Lichess doesn't have a direct rating-based endpoint
        # We'll use the daily puzzle for now
        # In the future, could fetch from puzzle database or use other endpoints
        return self.get_daily_puzzle()
    
    def parse_puzzle(self, puzzle_data: Dict) -> Optional[Dict]:
        """
        Parse Lichess puzzle data into a usable format.
        
        Args:
            puzzle_data: Raw puzzle data from Lichess API
            
        Returns:
            Parsed puzzle dictionary with:
            - initial_position: chess.Board
            - moves: List of UCI moves
            - solution: List of solution moves
            - rating: Puzzle rating
            - themes: List of themes
            - puzzle_id: Puzzle ID
        """
        try:
            puzzle_info = puzzle_data.get('puzzle', {})
            game_info = puzzle_data.get('game', {})
            
            # Parse initial position from FEN
            initial_fen = puzzle_info.get('initialPly', 0)
            
            # Get the solution moves (UCI format)
            solution_moves = puzzle_info.get('solution', [])
            
            # Get puzzle metadata
            rating = puzzle_info.get('rating', 1500)
            themes = puzzle_info.get('themes', [])
            puzzle_id = puzzle_info.get('id', 'unknown')
            
            # Reconstruct the board position from FEN if available
            board = chess.Board()
            
            # Try to get FEN directly from puzzle data first (most reliable)
            fen = puzzle_info.get('fen')
            if not fen and game_info:
                fen = game_info.get('fen')
            
            if fen:
                try:
                    board = chess.Board(fen)
                except Exception:
                    # If FEN parsing fails, try to reconstruct from PGN
                    pass
            
            # If no FEN, try to reconstruct from PGN
            if not fen or not board:
                game_moves_pgn = game_info.get('pgn', '') if game_info else ''
                initial_ply = puzzle_info.get('initialPly', 0)
                
                if game_moves_pgn:
                    try:
                        from io import StringIO
                        game = chess.pgn.read_game(StringIO(game_moves_pgn))
                        if game:
                            # Navigate to the puzzle position by playing moves
                            node = game
                            for _ in range(min(initial_ply, 200)):  # Limit to prevent infinite loops
                                if node.variations:
                                    node = node.variation(0)
                                else:
                                    break
                            if node:
                                board = node.board()
                    except Exception as e:
                        # If PGN parsing fails, use starting position
                        board = chess.Board()
            
            return {
                'initial_position': board,
                'moves': solution_moves,
                'solution': solution_moves,
                'rating': rating,
                'themes': themes,
                'puzzle_id': puzzle_id,
                'game_info': game_info
            }
        except Exception as e:
            st.error(f"Error parsing puzzle: {e}")
            return None


class PuzzlePlayer:
    """Interactive puzzle player with move validation."""
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """Initialize puzzle player."""
        self.stockfish_path = stockfish_path
        self.current_puzzle = None
        self.user_moves = []
        self.solution_index = 0
        self.puzzle_state = "waiting"  # waiting, playing, correct, wrong, solved
        self.board_gui = None
    
    def load_puzzle(self, puzzle_data: Dict):
        """
        Load a puzzle for solving.
        
        Args:
            puzzle_data: Parsed puzzle dictionary
        """
        self.current_puzzle = puzzle_data
        self.user_moves = []
        self.solution_index = 0
        self.puzzle_state = "playing"
        
        # Initialize board GUI if not already done
        if self.board_gui is None:
            from chess_board_gui import ChessBoardGUI
            self.board_gui = ChessBoardGUI()
    
    def get_current_board(self) -> Optional[chess.Board]:
        """Get the current board position after user's moves."""
        if not self.current_puzzle:
            return None
        
        board = self.current_puzzle['initial_position'].copy()
        for move_uci in self.user_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    return None  # Invalid move
            except Exception:
                return None
        
        return board
    
    def make_move(self, move_uci: str) -> Tuple[bool, str]:
        """
        Make a move in the puzzle.
        
        Args:
            move_uci: Move in UCI format (e.g., "e2e4")
            
        Returns:
            Tuple of (is_correct, message)
        """
        if not self.current_puzzle or self.puzzle_state != "playing":
            return False, "No puzzle loaded or puzzle already solved"
        
        solution_moves = self.current_puzzle['solution']
        
        if self.solution_index >= len(solution_moves):
            return False, "Puzzle already solved!"
        
        # Check if the move matches the solution
        expected_move = solution_moves[self.solution_index]
        
        if move_uci.lower() == expected_move.lower():
            # Correct move
            self.user_moves.append(move_uci)
            self.solution_index += 1
            
            # Check if puzzle is complete
            if self.solution_index >= len(solution_moves):
                self.puzzle_state = "solved"
                return True, "🎉 Puzzle solved! Excellent work!"
            else:
                # Move to next move in puzzle (opponent's turn or next part)
                # For puzzles, we typically need to play both our moves and opponent responses
                # The solution contains alternating moves
                return True, f"✅ Correct! Now play move {self.solution_index + 1}/{len(solution_moves)}"
        else:
            # Wrong move
            self.puzzle_state = "wrong"
            return False, f"❌ Incorrect. The correct move was {expected_move}"
    
    def get_hint(self) -> Optional[str]:
        """Get a hint for the current move (next move in solution)."""
        if not self.current_puzzle or self.solution_index >= len(self.current_puzzle['solution']):
            return None
        
        next_move = self.current_puzzle['solution'][self.solution_index]
        
        # Convert UCI to SAN for hint
        board = self.get_current_board()
        if board:
            try:
                move = chess.Move.from_uci(next_move)
                if move in board.legal_moves:
                    return f"Hint: Try {board.san(move)}"
            except Exception:
                pass
        
        return f"Hint: Try {next_move}"
    
    def reset_puzzle(self):
        """Reset the puzzle to the beginning."""
        if self.current_puzzle:
            self.user_moves = []
            self.solution_index = 0
            self.puzzle_state = "playing"
    
    def get_theme_display(self) -> str:
        """Get a formatted string of puzzle themes."""
        if not self.current_puzzle:
            return ""
        
        themes = self.current_puzzle.get('themes', [])
        theme_names = {
            'mateIn1': 'Mate in 1',
            'mateIn2': 'Mate in 2',
            'mateIn3': 'Mate in 3',
            'endgame': 'Endgame',
            'middlegame': 'Middlegame',
            'opening': 'Opening',
            'crushing': 'Crushing',
            'advancedPawn': 'Advanced Pawn',
            'attackingF2F7': 'Attacking f2/f7',
            'capturingDefender': 'Capturing Defender',
            'discoveredAttack': 'Discovered Attack',
            'doubleCheck': 'Double Check',
            'fork': 'Fork',
            'hangingPiece': 'Hanging Piece',
            'kingsideAttack': 'Kingside Attack',
            'pin': 'Pin',
            'queensideAttack': 'Queenside Attack',
            'sacrifice': 'Sacrifice',
            'skewer': 'Skewer',
            'trappedPiece': 'Trapped Piece',
            'underPromotion': 'Under Promotion',
            'zugzwang': 'Zugzwang'
        }
        
        display_themes = [theme_names.get(t, t) for t in themes[:5]]  # Show first 5
        return ", ".join(display_themes) if display_themes else "General"

