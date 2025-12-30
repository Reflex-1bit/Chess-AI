"""
Interactive Puzzle Player GUI

Provides a Streamlit interface for playing Lichess puzzles interactively.
"""

import streamlit as st
import streamlit.components.v1 as components
import chess
from typing import Optional
from lichess_puzzles import LichessPuzzleAPI, PuzzlePlayer
from chess_board_gui import ChessBoardGUI
from puzzle_recommender import PuzzleRecommender


class PuzzlePlayerGUI:
    """GUI for playing chess puzzles interactively."""
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """Initialize puzzle player GUI."""
        self.puzzle_api = LichessPuzzleAPI()
        self.puzzle_player = PuzzlePlayer(stockfish_path=stockfish_path)
        self.board_gui = ChessBoardGUI()
        self.recommender = PuzzleRecommender()
    
    def render_puzzle_interface(self):
        """Render the main puzzle playing interface."""
        st.header("🧩 Chess Puzzles from Lichess")
        st.markdown("Solve tactical puzzles to improve your chess skills!")
        
        # Initialize puzzle in session state
        puzzle_key = "current_puzzle_data"
        puzzle_state_key = "puzzle_player_state"
        
        if puzzle_state_key not in st.session_state:
            st.session_state[puzzle_state_key] = None
        
        # Get coaching data for recommendations
        moves_data = st.session_state.get('moves_data')
        predictions = st.session_state.get('predictions')
        insights = None
        
        # Try to get insights if model is available
        if st.session_state.get('model') and predictions and moves_data:
            try:
                from insight_generator import InsightGenerator
                insight_gen = InsightGenerator()
                insights = insight_gen.generate_insights(moves_data, predictions)
            except Exception:
                pass
        
        # Get AI-powered recommendations
        recommendations = self.recommender.get_puzzle_recommendations(
            moves_data=moves_data,
            predictions=predictions,
            insights=insights,
            user_rating=1500
        )
        
        # Sidebar for puzzle controls
        with st.sidebar:
            st.markdown("### 🎯 AI Recommendations")
            if recommendations['recommended_themes']:
                st.info(recommendations['recommendation_text'])
                st.markdown("**Recommended themes:**")
                for theme in recommendations['recommended_themes'][:3]:
                    st.markdown(f"- {theme}")
            else:
                st.info("Complete game analysis to get personalized puzzle recommendations!")
            
            st.markdown("---")
            st.markdown("### Puzzle Controls")
            
            col_rec, col_daily = st.columns(2)
            with col_rec:
                if st.button("🎯 Recommended", key="load_recommended", help="Load puzzle based on AI analysis", use_container_width=True):
                    with st.spinner("Fetching recommended puzzle..."):
                        puzzle_data = self.puzzle_api.get_daily_puzzle()
                        if puzzle_data:
                            parsed = self.puzzle_api.parse_puzzle(puzzle_data)
                            if parsed:
                                st.session_state[puzzle_key] = parsed
                                self.puzzle_player.load_puzzle(parsed)
                                st.session_state[puzzle_state_key] = self.puzzle_player.puzzle_state
                                st.success("✅ Recommended puzzle loaded!")
                                st.rerun()
                        else:
                            st.error("❌ Could not fetch puzzle. Please try again.")
            
            with col_daily:
                if st.button("📅 Daily", key="load_daily", help="Load today's daily puzzle", use_container_width=True):
                    with st.spinner("Fetching daily puzzle from Lichess..."):
                        puzzle_data = self.puzzle_api.get_daily_puzzle()
                        if puzzle_data:
                            parsed = self.puzzle_api.parse_puzzle(puzzle_data)
                            if parsed:
                                st.session_state[puzzle_key] = parsed
                                self.puzzle_player.load_puzzle(parsed)
                                st.session_state[puzzle_state_key] = self.puzzle_player.puzzle_state
                                st.success("✅ Puzzle loaded!")
                                st.rerun()
                        else:
                            st.error("❌ Could not fetch puzzle. Please try again.")
            
            if st.button("🔄 Load New Puzzle", key="load_puzzle", use_container_width=True):
                with st.spinner("Fetching puzzle from Lichess..."):
                    puzzle_data = self.puzzle_api.get_daily_puzzle()
                    if puzzle_data:
                        parsed = self.puzzle_api.parse_puzzle(puzzle_data)
                        if parsed:
                            st.session_state[puzzle_key] = parsed
                            self.puzzle_player.load_puzzle(parsed)
                            st.session_state[puzzle_state_key] = self.puzzle_player.puzzle_state
                            st.success("✅ Puzzle loaded!")
                            st.rerun()
                    else:
                        st.error("❌ Could not fetch puzzle. Please try again.")
            
            if st.button("🔁 Reset Puzzle", key="reset_puzzle", use_container_width=True):
                if puzzle_key in st.session_state:
                    self.puzzle_player.reset_puzzle()
                    st.session_state[puzzle_state_key] = self.puzzle_player.puzzle_state
                    st.rerun()
            
            # Load puzzle from session state if available
            if puzzle_key in st.session_state:
                puzzle_data = st.session_state[puzzle_key]
                self.puzzle_player.load_puzzle(puzzle_data)
                if st.session_state[puzzle_state_key]:
                    self.puzzle_player.puzzle_state = st.session_state[puzzle_state_key]
                    self.puzzle_player.solution_index = len(self.puzzle_player.user_moves)
                
                st.markdown("---")
                st.markdown("**Puzzle Info**")
                st.markdown(f"**Rating:** {puzzle_data.get('rating', 'N/A')}")
                st.markdown(f"**Themes:** {self.puzzle_player.get_theme_display()}")
                st.markdown(f"**Moves:** {len(puzzle_data.get('solution', []))}")
                st.markdown(f"**Progress:** {self.puzzle_player.solution_index}/{len(puzzle_data.get('solution', []))}")
        
        # Main puzzle area
        if puzzle_key not in st.session_state or not st.session_state[puzzle_key]:
            st.info("👆 Click 'Load New Puzzle' in the sidebar to start!")
            return
        
        puzzle_data = st.session_state[puzzle_key]
        self.puzzle_player.load_puzzle(puzzle_data)
        if st.session_state[puzzle_state_key]:
            self.puzzle_player.puzzle_state = st.session_state[puzzle_state_key]
            self.puzzle_player.solution_index = len(self.puzzle_player.user_moves)
        
        # Get current board position
        current_board = self.puzzle_player.get_current_board()
        
        if not current_board:
            st.error("Invalid board position. Please reset the puzzle.")
            return
        
        # Main layout
        col_board, col_controls = st.columns([7, 3])
        
        with col_board:
            # Display puzzle status
            if self.puzzle_player.puzzle_state == "solved":
                st.success("🎉 **Puzzle Solved!** Excellent work!")
            elif self.puzzle_player.puzzle_state == "wrong":
                st.error("❌ **Incorrect Move!** Try again or reset the puzzle.")
            
            # Render interactive draggable chess board
            board_html = self.board_gui.render_interactive_board(current_board, size=640)
            components.html(board_html, height=720, scrolling=False)
            
            # Puzzle instructions
            st.markdown("**Instructions:**")
            st.markdown("- Make moves using the UCI notation input below")
            st.markdown("- Format: 'e2e4' (from square to square)")
            st.markdown("- The puzzle will validate your moves against the solution")
        
        with col_controls:
            st.markdown("### Move Input")
            
            # Move input
            move_input_key = "puzzle_move_input"
            move_uci = st.text_input(
                "Enter move (UCI format):",
                key=move_input_key,
                placeholder="e.g., e2e4",
                help="Enter move in UCI format: from square to square (e.g., e2e4, g1f3)"
            )
            
            col_make, col_hint = st.columns(2)
            with col_make:
                if st.button("▶️ Make Move", key="make_move", use_container_width=True):
                    if move_uci:
                        is_correct, message = self.puzzle_player.make_move(move_uci.strip())
                        st.session_state[puzzle_state_key] = self.puzzle_player.puzzle_state
                        
                        if is_correct:
                            st.success(message)
                            if self.puzzle_player.puzzle_state == "solved":
                                # Clear input and rerun
                                st.session_state[move_input_key] = ""
                        else:
                            st.error(message)
                        st.rerun()
                    else:
                        st.warning("Please enter a move")
            
            with col_hint:
                if st.button("💡 Get Hint", key="get_hint", use_container_width=True):
                    hint = self.puzzle_player.get_hint()
                    if hint:
                        st.info(hint)
                    else:
                        st.warning("No hint available")
            
            st.markdown("---")
            st.markdown("### Quick Moves")
            st.markdown("Common moves (click to input):")
            
            # Quick move buttons for common moves
            quick_moves = [
                ("e2e4", "King's Pawn"),
                ("d2d4", "Queen's Pawn"),
                ("g1f3", "King's Knight"),
                ("f1c4", "Bishop to c4"),
                ("e1g1", "Kingside Castling"),
                ("e1c1", "Queenside Castling")
            ]
            
            for uci, label in quick_moves[:4]:  # Show first 4
                if st.button(label, key=f"quick_{uci}", use_container_width=True):
                    st.session_state[move_input_key] = uci
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### Solution")
            if st.button("👁️ Show Solution", key="show_solution", use_container_width=True):
                solution_moves = puzzle_data.get('solution', [])
                solution_text = " → ".join(solution_moves)
                st.code(solution_text)
            
            # Display current board info
            st.markdown("---")
            st.markdown("### Position Info")
            st.markdown(f"**Turn:** {'White' if current_board.turn else 'Black'}")
            st.markdown(f"**FEN:** `{current_board.fen()}`")
            
            if current_board.is_check():
                st.warning("⚠️ Check!")
            if current_board.is_checkmate():
                st.error("♟️ Checkmate!")
            if current_board.is_stalemate():
                st.info("🤝 Stalemate")
    
    def close(self):
        """Clean up resources."""
        if hasattr(self.puzzle_player, 'close'):
            self.puzzle_player.close()

