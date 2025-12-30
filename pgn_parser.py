"""
PGN Parser with Stockfish Evaluation Extraction

Parses PGN games and extracts per-move data with Stockfish evaluations.
Computes eval loss (ground truth labels) for each move.
"""

import chess
import chess.pgn
import chess.engine
from typing import List, Dict, Optional
import os
import re
import sys
import subprocess

# Fix for Windows asyncio subprocess issues in Python 3.14+
if sys.platform == 'win32':
    try:
        import asyncio
        # Use ProactorEventLoopPolicy on Windows to fix subprocess issues
        if sys.version_info >= (3, 14):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass  # If setting policy fails, continue anyway


class PGNParser:
    """Parses PGN files and extracts moves with Stockfish evaluations."""
    
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 15):
        """
        Initialize parser with Stockfish engine.
        
        Args:
            stockfish_path: Path to Stockfish executable. If None, tries to auto-detect.
            depth: Analysis depth for Stockfish (higher = more accurate, slower)
        """
        self.depth = depth
        
        # Auto-detect Stockfish if not provided
        if stockfish_path is None or stockfish_path.strip() == "":
            # Check for bundled Stockfish (when running as exe)
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = sys._MEIPASS
                bundled_stockfish = os.path.join(base_path, 'stockfish_bundled', 'stockfish.exe')
                if os.path.exists(bundled_stockfish):
                    stockfish_path = bundled_stockfish
                else:
                    stockfish_path = "stockfish"
            else:
                # Check for local bundled Stockfish
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Check multiple possible locations
                possible_paths = [
                    os.path.join(current_dir, 'stockfish', 'stockfish-windows-x86-64-avx2.exe'),
                    os.path.join(current_dir, 'stockfish', 'stockfish.exe'),
                    os.path.join(current_dir, 'stockfish_bundled', 'stockfish.exe'),
                ]
                stockfish_path = "stockfish"  # Default
                for path in possible_paths:
                    if os.path.exists(path):
                        stockfish_path = path
                        break
        
        # Normalize the path if it's a file path
        if stockfish_path and stockfish_path != "stockfish":
            stockfish_path = os.path.normpath(os.path.abspath(stockfish_path))
        
        self.stockfish_path = stockfish_path
        self.engine = None
        
    def _get_engine(self):
        """Lazy initialization of Stockfish engine."""
        if self.engine is None:
            # Validate path if not "stockfish" (which means it's in PATH)
            if self.stockfish_path != "stockfish":
                if not os.path.exists(self.stockfish_path):
                    raise RuntimeError(
                        f"Stockfish executable not found at: {self.stockfish_path}\n"
                        f"Please check that the path is correct and the file exists."
                    )
                if not os.path.isfile(self.stockfish_path):
                    raise RuntimeError(
                        f"Stockfish path is not a file: {self.stockfish_path}"
                    )
            
            try:
                # First verify the file exists and is accessible
                if not os.path.exists(self.stockfish_path):
                    raise FileNotFoundError(f"File not found: {self.stockfish_path}")
                
                # Normalize the path for Windows (handle backslashes properly)
                stockfish_path_normalized = os.path.normpath(os.path.abspath(self.stockfish_path))
                
                # Test if we can actually run the executable using subprocess first
                # This helps identify issues before python-chess tries to use it
                import subprocess
                try:
                    test_process = subprocess.Popen(
                        [stockfish_path_normalized],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    # Send uci command to verify it responds
                    test_process.stdin.write(b"uci\n")
                    test_process.stdin.flush()
                    test_process.stdin.write(b"quit\n")
                    test_process.stdin.flush()
                    test_process.wait(timeout=3)
                    if test_process.returncode is None:
                        test_process.terminate()
                        test_process.wait(timeout=1)
                    if test_process.returncode is None:
                        test_process.kill()
                except subprocess.TimeoutExpired:
                    test_process.kill()
                    raise RuntimeError(
                        f"Stockfish executable does not respond.\n"
                        f"Path: {stockfish_path_normalized}\n"
                        f"The executable may be corrupted or incompatible."
                    )
                except Exception as subprocess_error:
                    raise RuntimeError(
                        f"Cannot execute Stockfish executable.\n"
                        f"Path: {stockfish_path_normalized}\n"
                        f"Error: {subprocess_error}\n\n"
                        f"Possible causes:\n"
                        f"1. The executable is not compatible with your Windows version\n"
                        f"2. Missing required DLLs (try the regular version instead of AVX2)\n"
                        f"3. Antivirus blocking the process\n"
                        f"4. Try the non-AVX2 version from stockfishchess.org"
                    )
                
                # Now try to start the engine with python-chess
                # Use the command parameter which sometimes works better than just path
                try:
                    # Try with just the path first
                    self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path_normalized)
                except NotImplementedError as nie:
                    # This happens on Windows with Python 3.14+ due to asyncio subprocess issues
                    # Try to work around by using a different event loop policy
                    try:
                        import asyncio
                        # Try using WindowsProactorEventLoopPolicy for Windows
                        if sys.platform == 'win32':
                            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                            # Try again with the new policy
                            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path_normalized)
                        else:
                            raise nie
                    except Exception as policy_error:
                        # If event loop policy doesn't work, provide helpful error
                        raise RuntimeError(
                            f"Unable to start Stockfish engine (Windows asyncio issue).\n"
                            f"Path: {stockfish_path_normalized}\n"
                            f"This is a known issue with Python 3.14+ on Windows.\n\n"
                            f"Solutions:\n"
                            f"1. <strong>Use Python 3.11 or 3.12</strong> (recommended - avoids this issue)\n"
                            f"2. Use the non-AVX2 version: stockfish-windows-x86-64.exe\n"
                            f"3. Try running with: python -m asyncio\n"
                            f"4. Check if antivirus is blocking the executable\n\n"
                            f"Original error: {nie}"
                        )
                
                # Test if it responds
                try:
                    self.engine.ping()
                except Exception as ping_error:
                    try:
                        self.engine.quit()
                    except:
                        pass
                    raise RuntimeError(
                        f"Stockfish started but did not respond: {ping_error}\n"
                        f"Path: {stockfish_path_normalized}\n"
                        f"The executable may be corrupted, incompatible, or blocked by antivirus."
                    )
                    
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Failed to find Stockfish executable at: {self.stockfish_path}\n"
                    f"Please check that the path is correct and the file exists.\n"
                    f"Full error: {e}"
                )
            except PermissionError as e:
                raise RuntimeError(
                    f"Permission denied when trying to run Stockfish at: {self.stockfish_path}\n"
                    f"Try running the application as administrator or check file permissions.\n"
                    f"Full error: {e}"
                )
            except NotImplementedError as nie:
                raise RuntimeError(
                    f"Cannot start Stockfish executable on Windows.\n"
                    f"Path: {self.stockfish_path}\n"
                    f"Error: {nie}\n\n"
                    f"This is often caused by:\n"
                    f"1. Using AVX2 version when your CPU doesn't support it - try the regular version\n"
                    f"2. Missing Visual C++ runtime libraries\n"
                    f"3. Antivirus blocking subprocess execution\n\n"
                    f"Solutions:\n"
                    f"1. Download the regular (non-AVX2) version: stockfish-windows-x86-64.exe\n"
                    f"2. Install Visual C++ Redistributable from Microsoft\n"
                    f"3. Add the file to antivirus exclusions\n"
                    f"4. Try running Streamlit as administrator"
                )
            except Exception as e:
                error_type = type(e).__name__
                raise RuntimeError(
                    f"Failed to start Stockfish at: {self.stockfish_path}\n"
                    f"Error type: {error_type}\n"
                    f"Error message: {e}\n"
                    f"Make sure:\n"
                    f"1. The executable is valid and compatible with your system\n"
                    f"2. You have permission to run it\n"
                    f"3. Any antivirus software is not blocking it"
                )
        return self.engine
    
    def parse_game(self, pgn_file: str, player_name: Optional[str] = None) -> List[Dict]:
        """
        Parse a single PGN game and extract move data with evaluations.
        
        Args:
            pgn_file: Path to PGN file or PGN string
            player_name: Name of the player to analyze (if None, analyzes both players)
        
        Returns:
            List of move dictionaries, each containing:
            - move_number: Move number in game
            - move_san: Move in standard algebraic notation
            - board_before: Board state before move
            - board_after: Board state after move
            - eval_before: Stockfish evaluation before move (centipawns, from player's perspective)
            - eval_after: Stockfish evaluation after move (centipawns, from player's perspective)
            - eval_loss: eval_before - eval_after (positive = mistake)
            - is_white: Whether the player is white
            - game_phase: 'opening', 'middlegame', or 'endgame'
        """
        engine = self._get_engine()
        
        # Parse PGN
        try:
            if os.path.isfile(pgn_file):
                with open(pgn_file, 'r', encoding='utf-8') as f:
                    game = chess.pgn.read_game(f)
            else:
                game = chess.pgn.read_game(chess.io.StringIO(pgn_file))
        except Exception as e:
            # If parsing fails, return empty list
            return []
        
        if game is None:
            return []
        
        # Check if game has any moves
        try:
            mainline = list(game.mainline())
            if len(mainline) == 0:
                # Game has no moves (empty game, abandoned game, etc.)
                return []
        except Exception:
            # If we can't get mainline, return empty
            return []
        
        moves_data = []
        board = game.board()
        
        # Determine which player to analyze
        analyze_white = player_name is None or (
            game.headers.get("White", "").lower() == player_name.lower()
        )
        analyze_black = player_name is None or (
            game.headers.get("Black", "").lower() == player_name.lower()
        )
        
        move_number = 0
        for node in game.mainline():
            move = node.move
            is_white_move = board.turn == chess.WHITE
            
            # Only analyze moves for the specified player
            if (is_white_move and not analyze_white) or (not is_white_move and not analyze_black):
                board.push(move)
                continue
            
            # Get SAN notation before pushing the move (san() requires move to be legal from current board state)
            move_san = board.san(move)
            
            # Get evaluation before move
            info_before = engine.analyse(board, chess.engine.Limit(depth=self.depth))
            eval_before = self._info_to_centipawns(info_before, is_white_move)
            
            # Create board before move (copy current state)
            board_before = board.copy()
            
            # Make the move
            board.push(move)
            
            # Get evaluation after move
            info_after = engine.analyse(board, chess.engine.Limit(depth=self.depth))
            eval_after = self._info_to_centipawns(info_after, is_white_move)
            
            # Compute eval loss (from player's perspective)
            # Positive loss = player's position got worse
            eval_loss = eval_before - eval_after
            
            # Determine game phase
            game_phase = self._get_game_phase(board)
            
            # Extract additional metadata
            white_rating = game.headers.get("WhiteElo", "?")
            black_rating = game.headers.get("BlackElo", "?")
            time_control = game.headers.get("TimeControl", "")
            
            # Determine opponent rating
            if is_white_move:
                opponent_rating = int(black_rating) if black_rating.isdigit() else None
                player_rating = int(white_rating) if white_rating.isdigit() else None
            else:
                opponent_rating = int(white_rating) if white_rating.isdigit() else None
                player_rating = int(black_rating) if black_rating.isdigit() else None
            
            # Extract time remaining (if available in comments)
            time_remaining = None
            if node.comment:
                # Try to extract time from comment (format varies by site)
                time_match = re.search(r'(\d+\.?\d*)\s*(?:sec|s|second)', node.comment, re.IGNORECASE)
                if time_match:
                    time_remaining = float(time_match.group(1))
            
            move_data = {
                'move_number': move_number + 1,
                'move_san': move_san,
                'board_before': board_before,
                'board_after': board.copy(),
                'eval_before': eval_before,
                'eval_after': eval_after,
                'eval_loss': eval_loss,
                'is_white': is_white_move,
                'game_phase': game_phase,
                'opponent_rating': opponent_rating,
                'player_rating': player_rating,
                'time_remaining': time_remaining,
                'time_control': time_control,
            }
            
            moves_data.append(move_data)
            move_number += 1
        
        return moves_data
    
    def parse_games_directory(self, games_dir: str, player_name: Optional[str] = None) -> List[Dict]:
        """
        Parse all PGN files in a directory.
        
        Args:
            games_dir: Directory containing PGN files
            player_name: Name of the player to analyze
        
        Returns:
            List of all move dictionaries from all games
        """
        all_moves = []
        
        for filename in os.listdir(games_dir):
            if filename.endswith('.pgn'):
                filepath = os.path.join(games_dir, filename)
                try:
                    moves = self.parse_game(filepath, player_name)
                    all_moves.extend(moves)
                    print(f"Parsed {filename}: {len(moves)} moves")
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")
        
        return all_moves
    
    def _info_to_centipawns(self, info: Dict, is_white: bool) -> float:
        """
        Convert Stockfish info to centipawns from player's perspective.
        
        Args:
            info: Stockfish analysis info
            is_white: Whether the player is white
        
        Returns:
            Evaluation in centipawns (positive = better for player)
        """
        score = info.get("score")
        if score is None:
            return 0.0
        
        # Handle PovScore object - get the relative score
        relative_score = score.relative if hasattr(score, 'relative') else score
        
        # Convert to centipawns
        if relative_score.is_mate():
            # Mate in N moves - convert to large centipawn value
            mate_score = relative_score.mate()
            if mate_score > 0:
                cp = 10000  # Winning
            else:
                cp = -10000  # Losing
        else:
            # Get the centipawn value - try different methods
            if hasattr(relative_score, 'score'):
                cp = relative_score.score()
            elif hasattr(relative_score, 'cp'):
                cp = relative_score.cp
            else:
                # Fallback: convert using white's perspective
                white_score = score.pov(chess.WHITE) if hasattr(score, 'pov') else relative_score
                if hasattr(white_score, 'score'):
                    cp = white_score.score()
                elif hasattr(white_score, 'cp'):
                    cp = white_score.cp
                else:
                    # Last resort: try to access as int
                    cp = int(relative_score) if hasattr(relative_score, '__int__') else 0
        
        # The score is already relative to the side to move, so we need to flip for black
        if not is_white:
            cp = -cp
        
        return cp
    
    def _get_game_phase(self, board: chess.Board) -> str:
        """
        Determine game phase based on material.
        
        Returns:
            'opening', 'middlegame', or 'endgame'
        """
        # Count pieces (excluding pawns and kings)
        piece_map = board.piece_map()
        piece_count = 0
        for piece in piece_map.values():
            # Count only non-pawn, non-king pieces
            if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
                piece_count += 1
        
        if piece_count >= 20:
            return 'opening'
        elif piece_count >= 10:
            return 'middlegame'
        else:
            return 'endgame'
    
    def close(self):
        """Close Stockfish engine."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

