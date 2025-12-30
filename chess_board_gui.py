"""
Chess Board GUI Component for Streamlit

Provides interactive chess board visualization using HTML/CSS/JavaScript.
"""

import streamlit as st
import streamlit.components.v1 as components
import chess
try:
    import chess.svg
    HAS_SVG = True
except ImportError:
    HAS_SVG = False
from typing import Optional, List, Dict, Tuple
import base64
from io import BytesIO
from chess_board_visualizer import render_board_html_fallback


class ChessBoardGUI:
    """Interactive chess board visualization for Streamlit."""
    
    def __init__(self):
        """Initialize the chess board GUI."""
        pass
    
    def format_eval(self, cp: float) -> str:
        """Format evaluation in a user-friendly way with Stockfish evaluation terms."""
        if abs(cp) >= 10000:
            return "Checkmate"
        # Convert to pawns for better readability
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
    
    def render_board(self, board: chess.Board, 
                     last_move: Optional[chess.Move] = None,
                     highlighted_squares: Optional[List[int]] = None,
                     arrows: Optional[List[Tuple[int, int]]] = None,
                     size: int = 640) -> str:
        """
        Render a chess board as HTML.
        
        Args:
            board: Chess board position
            last_move: Last move played (to highlight)
            highlighted_squares: List of square indices to highlight
            arrows: List of (from_square, to_square) tuples for arrows
            size: Board size in pixels
        
        Returns:
            HTML string with embedded SVG
        """
        if HAS_SVG:
            # Generate SVG arrows
            svg_arrows = arrows if arrows else []
            if last_move:
                svg_arrows.append((last_move.from_square, last_move.to_square))
            
            squares = highlighted_squares if highlighted_squares else None
            
            try:
                svg = chess.svg.board(
                    board=board,
                    arrows=svg_arrows if svg_arrows else None,
                    squares=squares,
                    size=size
                )
                
                # Convert SVG to base64 for embedding
                svg_bytes = svg.encode('utf-8')
                svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
                
                # Create HTML with embedded SVG
                html = f"""
                <div style="display: flex; justify-content: center; margin: 20px 0;">
                    <img src="data:image/svg+xml;base64,{svg_b64}" 
                         style="max-width: 100%; height: auto; border: 2px solid #333; border-radius: 4px;" />
                </div>
                """
                
                return html
            except Exception:
                # Fall through to HTML fallback
                pass
        
        # Fallback: Use HTML/CSS board visualization (with drag-and-drop if needed)
        return render_board_html_fallback(board, size, interactive=False)
    
    def render_interactive_board(self, board: chess.Board, size: int = 640) -> str:
        """
        Render an interactive draggable chess board.
        
        Args:
            board: Chess board position
            size: Board size in pixels
            
        Returns:
            HTML string with interactive draggable board
        """
        return render_board_html_fallback(board, size, interactive=True)
    
    def build_full_game_sequence(self, moves_data: List[Dict]) -> List[Dict]:
        """Build a sequence showing all moves (both players) from parsed moves."""
        if not moves_data:
            return []
        
        # Start with initial position
        first_move = moves_data[0]
        board = chess.Board()
        full_sequence = []
        
        # We need to reconstruct the full game including opponent moves
        # For now, we'll show only the analyzed player's moves but indicate both sides
        move_number_display = 1
        
        for move_data in moves_data:
            board_before = move_data.get('board_before')
            board_after = move_data.get('board_after')
            is_white = move_data.get('is_white', True)
            move_san = move_data.get('move_san', '')
            
            if board_after:
                full_sequence.append({
                    'move_number': move_number_display,
                    'move_san': move_san,
                    'is_white': is_white,
                    'board_before': board_before if board_before else board,
                    'board_after': board_after,
                    'eval_before': move_data.get('eval_before', 0),
                    'eval_after': move_data.get('eval_after', 0),
                    'eval_loss': move_data.get('eval_loss', 0),
                    'game_phase': move_data.get('game_phase', 'unknown')
                })
                board = board_after
                move_number_display += 1
        
        return full_sequence
    
    def render_move_sequence(self, moves_data: List[Dict], 
                            all_moves: Optional[List[Dict]] = None,
                            current_move_idx: Optional[int] = None,
                            stockfish_path: Optional[str] = None,
                            analysis_depth: int = 15) -> Optional[int]:
        """
        Render a sequence of moves with interactive board.
        Shows all moves of both players.
        
        Args:
            moves_data: List of move dictionaries with board states (analyzed player's moves)
            all_moves: Optional full game sequence (if available)
            current_move_idx: Currently selected move index
        
        Returns:
            Selected move index
        """
        if not moves_data:
            st.warning("No moves to display")
            return None
        
        # Build full sequence (for now just use moves_data, but format to show both sides)
        sequence = all_moves if all_moves else self.build_full_game_sequence(moves_data)
        
        if not sequence:
            st.warning("No moves to display")
            return None
        
        # Initialize move index
        if current_move_idx is None:
            current_move_idx = len(sequence) - 1
        current_move_idx = max(0, min(current_move_idx, len(sequence) - 1))
        
        # Navigation buttons - use session state to prevent unnecessary reruns
        nav_key = f"nav_{id(sequence)}"
        if nav_key not in st.session_state:
            st.session_state[nav_key] = current_move_idx
        
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            prev_clicked = st.button("◀ Previous", key=f"prev_{nav_key}", disabled=(st.session_state[nav_key] == 0))
            if prev_clicked:
                st.session_state[nav_key] = max(0, st.session_state[nav_key] - 1)
                st.rerun()
        with col_info:
            st.markdown(f"**Move {st.session_state[nav_key] + 1} of {len(sequence)}** (Use Previous/Next buttons)")
        with col_next:
            next_clicked = st.button("Next ▶", key=f"next_{nav_key}", disabled=(st.session_state[nav_key] >= len(sequence) - 1))
            if next_clicked:
                st.session_state[nav_key] = min(len(sequence) - 1, st.session_state[nav_key] + 1)
                st.rerun()
        
        # Update current_move_idx from session state
        current_move_idx = st.session_state[nav_key]
        
        # Initialize variables
        current_eval = 0
        move_data = None
        if current_move_idx < len(sequence):
            move_data = sequence[current_move_idx]
        
        # Main content area - board on left, eval bar and move list on right
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if move_data:
                board_after = move_data['board_after']
                move_san = move_data['move_san']
                move_number = move_data['move_number']
                is_white = move_data['is_white']
                
                # Get the move that was made
                board_before = move_data.get('board_before')
                last_move = None
                if board_before:
                    # Find the move that was played
                    for move in board_before.legal_moves:
                        test_board = board_before.copy()
                        test_board.push(move)
                        if test_board.fen() == board_after.fen():
                            last_move = move
                            break
                
                # Generate arrows for tactical ideas (check, captures, threats)
                arrows = []
                if board_after:
                    # Add arrow for the move played
                    if last_move:
                        arrows.append((last_move.from_square, last_move.to_square))
                    
                    # Check for checks
                    if board_after.is_check():
                        king_square = board_after.king(not board_after.turn)
                        if king_square is not None:
                            # Find pieces giving check - iterate through all 64 squares
                            for square in range(64):
                                piece = board_after.piece_at(square)
                                if piece and piece.color == board_after.turn:
                                    if king_square in board_after.attacks(square):
                                        arrows.append((square, king_square))
                
                # Render board with better styling (larger)
                board_html = self.render_board(
                    board_after, 
                    last_move=last_move,
                    arrows=arrows if arrows else None,
                    size=640
                )
                components.html(board_html, height=720, scrolling=False)
                
                # Move info with better formatting
                color_name = "White" if is_white else "Black"
                st.markdown(f"**Move {move_number}:** {move_san} ({color_name})")
                
                eval_before = move_data.get('eval_before', 0)
                eval_after = move_data.get('eval_after', 0)
                eval_loss = move_data.get('eval_loss', 0)
                
                st.markdown(f"**Evaluation Before:** {self.format_eval(eval_before)}")
                st.markdown(f"**Evaluation After:** {self.format_eval(eval_after)}")
                
                if 'eval_loss' in move_data:
                    eval_loss_text = self.format_eval_loss(eval_loss)
                    st.markdown(f"**Loss:** You {eval_loss_text}")
                    
                    # Blunder indicator
                    if eval_loss > 200:
                        st.error(f"💥 **BLUNDER!** You {eval_loss_text}")
                    elif eval_loss > 100:
                        st.warning(f"⚠️ **Mistake:** You {eval_loss_text}")
                    elif eval_loss > 50:
                        st.info(f"⚡ **Inaccuracy:** You {eval_loss_text}")
                
                st.markdown(f"**Game Phase:** {move_data.get('game_phase', 'unknown').capitalize()}")
                
                # Use stored evaluation instead of running Stockfish every time (much faster!)
                current_eval = move_data.get('eval_after', 0)
                
                # Optional: Button to run deeper Stockfish analysis
                if stockfish_path and board_after:
                    if st.button("🔍 Run Deep Analysis", key=f"deep_analysis_{current_move_idx}"):
                        try:
                            import chess.engine
                            with st.spinner("Analyzing position..."):
                                with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                                    info = engine.analyse(board_after, chess.engine.Limit(depth=analysis_depth))
                                    score = info.get("score")
                                    
                                    if score:
                                        if hasattr(score, 'relative'):
                                            score = score.relative
                                        
                                        if score.is_mate():
                                            mate_in = score.mate()
                                            eval_text = f"Mate in {abs(mate_in)}" if mate_in else "Checkmate"
                                            st.info(f"**Live Evaluation:** {eval_text}")
                                            current_eval = 10000 if mate_in > 0 else -10000
                                        else:
                                            current_eval = score.score()
                                            eval_display = self.format_eval(current_eval)
                                            st.info(f"**Live Evaluation:** {eval_display}")
                                        
                                        # Show best move
                                        pv = info.get("pv", [])
                                        if pv:
                                            best_move = pv[0]
                                            best_move_san = board_after.san(best_move)
                                            st.success(f"**Best Move:** {best_move_san}")
                        except Exception as e:
                            st.warning(f"Could not analyze position: {e}")
        
        with col2:
            # Evaluation bar
            st.subheader("📊 Evaluation")
            
            # Get current evaluation from stored move data (fast, no Stockfish needed)
            if move_data:
                current_eval = move_data.get('eval_after', 0)
            else:
                current_eval = 0
            
            # Create eval bar (range: -1000 to +1000 centipawns, clipped to ±500 for display)
            eval_clipped = max(-500, min(500, current_eval))
            eval_percentage = ((eval_clipped + 500) / 1000) * 100  # Convert to 0-100%
            
            # Color based on evaluation
            if eval_clipped > 50:
                bar_color = "#22c55e"  # Green (advantage)
            elif eval_clipped < -50:
                bar_color = "#ef4444"  # Red (disadvantage)
            else:
                bar_color = "#94a3b8"  # Gray (equal)
            
            # Vertical eval bar
            eval_bar_html = f"""
            <div style="width: 60px; height: 400px; background-color: #e5e7eb; border: 2px solid #374151; border-radius: 8px; position: relative; margin: 20px auto;">
                <div style="position: absolute; bottom: 0; width: 100%; height: {eval_percentage}%; background-color: {bar_color}; border-radius: 0 0 6px 6px; transition: height 0.3s;">
                </div>
                <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: #1f2937; font-weight: bold; font-size: 14px; text-shadow: 1px 1px 2px white;">
                    {self.format_eval(current_eval)}
                </div>
            </div>
            """
            st.markdown(eval_bar_html, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Move List")
            st.markdown(f"*Showing all {len(sequence)} moves*")
            
            # Move selector dropdown - use session state to prevent unnecessary reruns
            selector_key = f"selector_{nav_key}"
            if selector_key not in st.session_state:
                st.session_state[selector_key] = current_move_idx
            
            selected_idx = st.selectbox(
                "Jump to Move",
                options=range(len(sequence)),
                format_func=lambda x: f"{x+1}. {sequence[x]['move_san']} ({'W' if sequence[x]['is_white'] else 'B'})",
                index=st.session_state[selector_key],
                key=selector_key
            )
            
            # Only update if selection changed
            if selected_idx != st.session_state[selector_key]:
                st.session_state[selector_key] = selected_idx
                st.session_state[nav_key] = selected_idx
                st.rerun()
            
            # Sync selector with nav
            if st.session_state[nav_key] != st.session_state[selector_key]:
                st.session_state[selector_key] = st.session_state[nav_key]
            
            # Display move list with colors (scrollable)
            st.markdown("**All Moves:**")
            
            # Create a scrollable container
            move_list_html = '<div style="max-height: 500px; overflow-y: auto; padding: 10px;">'
            for i, move_data in enumerate(sequence):
                eval_loss = move_data.get('eval_loss', 0)
                move_san = move_data['move_san']
                move_num = move_data['move_number']
                is_white = move_data.get('is_white', True)
                color_indicator = "W" if is_white else "B"
                
                # Color code by eval loss
                if eval_loss > 200:
                    emoji = "🔴"
                    style = "color: #dc2626; font-weight: bold;"
                elif eval_loss > 100:
                    emoji = "🟡"
                    style = "color: #d97706; font-weight: bold;"
                elif eval_loss > 50:
                    emoji = "🟠"
                    style = "color: #f59e0b;"
                else:
                    emoji = "🟢"
                    style = "color: #059669;"
                
                if i == current_move_idx:
                    style += " background-color: #f3f4f6; padding: 4px; border-radius: 4px; border-left: 3px solid #3b82f6;"
                
                if 'eval_loss' in move_data and move_data['eval_loss'] > 0:
                    eval_loss_text = self.format_eval_loss(move_data['eval_loss'])
                    eval_str = f"- {eval_loss_text}"
                else:
                    eval_str = ""
                move_list_html += f'<div style="{style}">{emoji} {move_num}. {move_san} ({color_indicator}) {eval_str}</div>'
            
            move_list_html += '</div>'
            st.markdown(move_list_html, unsafe_allow_html=True)
        
        return current_move_idx
