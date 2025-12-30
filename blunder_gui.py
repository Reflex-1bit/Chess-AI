"""
Chess.com-style Blunder Analysis Interface

Provides a modern interface similar to chess.com for analyzing blunders
with live Stockfish evaluation.
"""

import streamlit as st
import streamlit.components.v1 as components
import chess
import chess.engine
from typing import List, Dict, Optional, Tuple
import numpy as np
from chess_board_gui import ChessBoardGUI
from blunder_analyzer import BlunderAnalyzer


class BlunderAnalysisGUI:
    """Chess.com-style interface for blunder analysis."""
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """Initialize the blunder analysis GUI."""
        self.stockfish_path = stockfish_path
        self.board_gui = ChessBoardGUI()
        self.engine = None
        self.blunder_analyzer = BlunderAnalyzer(stockfish_path=stockfish_path)
    
    def _get_engine(self):
        """Get Stockfish engine for analysis."""
        if self.engine is None and self.stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception:
                pass
        return self.engine
    
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
    
    def _classify_move_quality(self, eval_loss: float) -> Tuple[str, str, str]:
        """
        Classify move quality based on eval loss.
        Returns: (emoji, label, color)
        """
        if eval_loss < 20:
            return ("⭐", "Best", "#22c55e")  # Green
        elif eval_loss < 50:
            return ("👍", "Excellent", "#3b82f6")  # Blue
        elif eval_loss < 100:
            return ("!", "Great", "#eab308")  # Yellow
        elif eval_loss < 200:
            return ("?", "Mistake", "#f59e0b")  # Orange
        elif eval_loss < 300:
            return ("??", "Blunder", "#ef4444")  # Red
        else:
            return ("💥", "Critical Blunder", "#dc2626")  # Dark Red
    
    def _build_full_game_sequence(self, moves_data: List[Dict]) -> List[Dict]:
        """Build complete game sequence with all moves."""
        if not moves_data:
            return []
        
        sequence = []
        board = chess.Board()
        move_number_display = 1
        
        for move_data in moves_data:
            board_after = move_data.get('board_after')
            board_before = move_data.get('board_before')
            is_white = move_data.get('is_white', True)
            move_san = move_data.get('move_san', '')
            eval_loss = move_data.get('eval_loss', 0)
            
            if board_after:
                sequence.append({
                    'move_number': move_number_display,
                    'move_san': move_san,
                    'is_white': is_white,
                    'board_before': board_before if board_before else board,
                    'board_after': board_after,
                    'eval_before': move_data.get('eval_before', 0),
                    'eval_after': move_data.get('eval_after', 0),
                    'eval_loss': eval_loss,
                    'game_phase': move_data.get('game_phase', 'unknown')
                })
                board = board_after
                move_number_display += 1
        
        return sequence
    
    def _render_eval_bar(self, current_eval: float, height: int = 400) -> str:
        """Render a vertical evaluation bar."""
        # Clip evaluation for display (range: -500 to +500 centipawns)
        eval_clipped = max(-500, min(500, current_eval))
        eval_percentage = ((eval_clipped + 500) / 1000) * 100  # Convert to 0-100%
        
        # Color based on evaluation
        if eval_clipped > 50:
            bar_color = "#22c55e"  # Green (advantage)
        elif eval_clipped < -50:
            bar_color = "#ef4444"  # Red (disadvantage)
        else:
            bar_color = "#94a3b8"  # Gray (equal)
        
        eval_text = self.format_eval(current_eval)
        
        return f"""
        <div style="width: 60px; height: {height}px; background-color: #e5e7eb; border: 2px solid #374151; border-radius: 8px; position: relative; margin: 20px auto;">
            <div style="position: absolute; bottom: 0; width: 100%; height: {eval_percentage}%; background-color: {bar_color}; border-radius: 0 0 6px 6px; transition: height 0.3s;">
            </div>
            <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: #1f2937; font-weight: bold; font-size: 14px; text-shadow: 1px 1px 2px white;">
                {eval_text}
            </div>
        </div>
        """
    
    def render_analysis_interface(self, moves_data: List[Dict], 
                                   blunder_threshold: float = 200.0,
                                   opening_name: Optional[str] = None,
                                   depth: int = 15,
                                   game_key: str = "blunder_analysis"):
        """
        Render the main analysis interface similar to chess.com.
        
        Args:
            moves_data: List of move dictionaries
            blunder_threshold: Threshold for blunder detection
            opening_name: Name of the opening (if known)
            depth: Stockfish analysis depth
            game_key: Unique key for session state
        """
        if not moves_data:
            st.warning("No moves to display")
            return
        
        # Build full game sequence
        sequence = self._build_full_game_sequence(moves_data)
        if not sequence:
            st.warning("No moves to display")
            return
        
        # Initialize navigation state
        nav_key = f"nav_{game_key}"
        if nav_key not in st.session_state:
            st.session_state[nav_key] = len(sequence) - 1  # Start at last move
        
        current_idx = st.session_state[nav_key]
        current_idx = max(0, min(current_idx, len(sequence) - 1))
        
        # Main layout: Board on left (70%), Sidebar on right (30%)
        col_board, col_sidebar = st.columns([7, 3])
        
        with col_board:
            # Chess board area
            if current_idx < len(sequence):
                move_data = sequence[current_idx]
                board_after = move_data['board_after']
                board_before = move_data.get('board_before')
                
                # Get the move that was played for highlighting
                last_move = None
                if board_before:
                    for move in board_before.legal_moves:
                        test_board = board_before.copy()
                        test_board.push(move)
                        if test_board.fen() == board_after.fen():
                            last_move = move
                            break
                
                # Generate arrows for tactical ideas
                arrows = []
                if last_move:
                    arrows.append((last_move.from_square, last_move.to_square))
                
                # Render board (larger size)
                board_html = self.board_gui.render_board(
                    board_after,
                    last_move=last_move,
                    arrows=arrows if arrows else None,
                    size=640
                )
                components.html(board_html, height=720, scrolling=False)
                
                # Move information below board
                move_san = move_data['move_san']
                move_number = move_data['move_number']
                is_white = move_data['is_white']
                color_name = "White" if is_white else "Black"
                
                st.markdown(f"**Move {move_number}:** {move_san} ({color_name})")
                
                # Show blunder explanation if this is a blunder
                eval_loss = move_data.get('eval_loss', 0)
                if eval_loss > 200:  # Blunder threshold
                    st.markdown("---")
                    explanation = self.blunder_analyzer._explain_blunder(move_data, eval_loss)
                    if explanation:
                        st.error("💥 **Blunder Detected**")
                        st.markdown(explanation)
                    
                    # Show blunder points
                    eval_loss_text = self.format_eval_loss(eval_loss)
                    st.warning(f"Lost {eval_loss_text} of Stockfish evaluation.")
        
        with col_sidebar:
            # Sidebar content
            st.markdown("### Analysis")
            
            # Stockfish engine info
            if self.stockfish_path:
                st.markdown("**Engine:** Stockfish")
                st.markdown(f"**Depth:** {depth}")
            
            # Opening name
            if opening_name:
                st.markdown(f"**Opening:** {opening_name}")
            
            st.markdown("---")
            
            # Live evaluation bar - run Stockfish analysis automatically
            current_eval = move_data.get('eval_after', 0) if current_idx < len(sequence) else 0
            stored_eval = current_eval
            
            # Cache key for live evaluation (use FEN to cache by position)
            live_eval_cache_key = f"live_eval_cache_{game_key}"
            if live_eval_cache_key not in st.session_state:
                st.session_state[live_eval_cache_key] = {}
            
            eval_cache = st.session_state[live_eval_cache_key]
            position_fen = sequence[current_idx]['board_after'].fen() if current_idx < len(sequence) else ""
            
            # Get live evaluation from Stockfish if available
            if self._get_engine() and current_idx < len(sequence) and position_fen:
                # Check cache first
                if position_fen in eval_cache:
                    current_eval = eval_cache[position_fen]
                else:
                    # Run analysis and cache result
                    try:
                        board_for_eval = sequence[current_idx]['board_after']
                        with st.spinner("Analyzing position..."):
                            info = self.engine.analyse(board_for_eval, chess.engine.Limit(depth=depth))
                            score = info.get("score")
                            if score:
                                if hasattr(score, 'relative'):
                                    score = score.relative
                                if score.is_mate():
                                    mate_in = score.mate()
                                    current_eval = 10000 if mate_in > 0 else -10000
                                else:
                                    current_eval = score.score()
                                # Cache the result
                                eval_cache[position_fen] = current_eval
                                st.session_state[live_eval_cache_key] = eval_cache
                    except Exception as e:
                        # If analysis fails, use stored eval
                        pass
            
            st.markdown("**Evaluation**")
            eval_bar_html = self._render_eval_bar(current_eval)
            st.markdown(eval_bar_html, unsafe_allow_html=True)
            
            # Show eval info
            if current_eval != stored_eval:
                st.caption(f"Stored: {self.format_eval(stored_eval)} | Live: {self.format_eval(current_eval)}")
            else:
                st.caption(f"Eval: {self.format_eval(current_eval)}")
            
            st.markdown("---")
            
            # Move quality summary
            move_quality_counts = {'Best': 0, 'Excellent': 0, 'Great': 0, 'Mistake': 0, 'Blunder': 0, 'Critical Blunder': 0}
            for move in sequence:
                _, label, _ = self._classify_move_quality(move.get('eval_loss', 0))
                if label in move_quality_counts:
                    move_quality_counts[label] += 1
            
            st.markdown("**Move Quality**")
            if move_quality_counts['Best'] > 0:
                st.markdown(f"⭐ {move_quality_counts['Best']} Best")
            if move_quality_counts['Excellent'] > 0:
                st.markdown(f"👍 {move_quality_counts['Excellent']} Excellent")
            if move_quality_counts['Great'] > 0:
                st.markdown(f"! {move_quality_counts['Great']} Great")
            
            st.markdown("---")
            
            # Navigation controls - better formatted buttons
            st.markdown("**Navigation**")
            nav_cols = st.columns(4)
            with nav_cols[0]:
                if st.button("⏮️ Start", key=f"start_{game_key}", help="Go to start", use_container_width=True):
                    st.session_state[nav_key] = 0
                    st.rerun()
            with nav_cols[1]:
                if st.button("◀️ Previous", key=f"prev_{game_key}", disabled=(current_idx == 0), help="Previous move", use_container_width=True):
                    st.session_state[nav_key] = max(0, current_idx - 1)
                    st.rerun()
            with nav_cols[2]:
                if st.button("Next ▶️", key=f"next_{game_key}", disabled=(current_idx >= len(sequence) - 1), help="Next move", use_container_width=True):
                    st.session_state[nav_key] = min(len(sequence) - 1, current_idx + 1)
                    st.rerun()
            with nav_cols[3]:
                if st.button("End ⏭️", key=f"end_{game_key}", help="Go to end", use_container_width=True):
                    st.session_state[nav_key] = len(sequence) - 1
                    st.rerun()
            
            st.markdown("---")
            
            # Utility buttons - better formatted
            st.markdown("**Tools**")
            util_cols = st.columns(3)
            with util_cols[0]:
                if st.button("🔄 Flip", key=f"flip_{game_key}", help="Flip board", use_container_width=True):
                    st.info("Flip board feature - coming soon")
            with util_cols[1]:
                if st.button("📋 FEN", key=f"fen_{game_key}", help="Copy FEN", use_container_width=True):
                    if current_idx < len(sequence):
                        fen = sequence[current_idx]['board_after'].fen()
                        st.code(fen)
            with util_cols[2]:
                if st.button("📄 PGN", key=f"pgn_{game_key}", help="Copy PGN", use_container_width=True):
                    st.info("PGN export - coming soon")
            
            st.markdown("---")
            
            # Move list (scrollable)
            st.markdown("**Move List**")
            st.markdown(f"*Showing {len(sequence)} moves*")
            
            # Move selector dropdown
            selected_move_idx = st.selectbox(
                "Jump to Move",
                options=range(len(sequence)),
                format_func=lambda x: f"{x+1}. {sequence[x]['move_san']} ({'W' if sequence[x]['is_white'] else 'B'})",
                index=current_idx,
                key=f"move_selector_{game_key}"
            )
            
            if selected_move_idx != current_idx:
                st.session_state[nav_key] = selected_move_idx
                st.rerun()
            
            # Display move list
            st.markdown("**All Moves:**")
            move_list_html = '<div style="max-height: 400px; overflow-y: auto; padding: 10px; background-color: #f9fafb; border-radius: 4px;">'
            
            for i, move_item in enumerate(sequence):
                move_san = move_item['move_san']
                move_num = move_item['move_number']
                is_white = move_item['is_white']
                eval_loss = move_item.get('eval_loss', 0)
                color_indicator = "W" if is_white else "B"
                
                # Get move quality
                emoji, label, color = self._classify_move_quality(eval_loss)
                
                # Highlight current move
                if i == current_idx:
                    style = f"background-color: #dbeafe; padding: 6px; border-radius: 4px; border-left: 4px solid #3b82f6; margin: 2px 0; font-weight: bold;"
                else:
                    style = f"padding: 4px; margin: 2px 0; color: {color};"
                
                eval_loss_text = self.format_eval_loss(eval_loss) if eval_loss > 0 else ""
                eval_str = f"({eval_loss_text})" if eval_loss_text else ""
                move_list_html += f'<div style="{style}">{emoji} {move_num}. {move_san} ({color_indicator}) {eval_str}</div>'
            
            move_list_html += '</div>'
            st.markdown(move_list_html, unsafe_allow_html=True)
            
            # Blunder highlights
            blunders = [i for i, m in enumerate(sequence) if m.get('eval_loss', 0) > blunder_threshold]
            if blunders:
                st.markdown("---")
                st.markdown("**💥 Blunders**")
                for blunder_idx in blunders:
                    blunder_move = sequence[blunder_idx]
                    eval_loss = blunder_move.get('eval_loss', 0)
                    eval_loss_text = self.format_eval_loss(eval_loss)
                    if st.button(f"Move {blunder_move['move_number']}: {blunder_move['move_san']} ({eval_loss_text})", 
                                key=f"blunder_{blunder_idx}_{game_key}"):
                        st.session_state[nav_key] = blunder_idx
                        st.rerun()
    
    def close(self):
        """Close the engine."""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None

