"""
Streamlit Web App for Chess Coaching System

Interactive web interface for training models and generating insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from pathlib import Path
import time

from pgn_parser import PGNParser
from feature_extractor import FeatureExtractor
from model_trainer import ChessMistakeModel
from insight_generator import InsightGenerator
from visualizations import ChessVisualizer
from chess_board_gui import ChessBoardGUI
from blunder_analyzer import BlunderAnalyzer
from blunder_gui import BlunderAnalysisGUI
from opening_database import OpeningDatabase
from chesscom_api import ChessComAPI
from puzzle_player_gui import PuzzlePlayerGUI

# Page configuration
st.set_page_config(
    page_title="Chess Coaching System",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No custom CSS - use Streamlit's default styling which handles text colors properly

# Initialize session state
if 'moves_data' not in st.session_state:
    st.session_state.moves_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None
if 'stockfish_path' not in st.session_state:
    st.session_state.stockfish_path = None


def main():
    # Simple header
    st.title("♟️ Chess Coaching System")
    st.markdown("Human Mistake Modeling for Personalized Chess Coaching")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stockfish path configuration
        st.subheader("🔧 Stockfish Configuration")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check for Stockfish in local stockfish folder
        stockfish_folder = os.path.join(current_dir, "stockfish")
        possible_stockfish = []
        if os.path.exists(stockfish_folder):
            for file in os.listdir(stockfish_folder):
                if file.endswith(".exe"):
                    possible_stockfish.append(os.path.join(stockfish_folder, file))
        
        # File uploader for Stockfish executable
        st.markdown("**Upload Stockfish Executable**")
        uploaded_stockfish = st.file_uploader(
            "Upload Stockfish executable",
            type=['exe'],
            help="Upload the Stockfish .exe file",
            key="stockfish_uploader"
        )
        
        if uploaded_stockfish is not None:
            # Save uploaded file
            uploads_dir = os.path.join(current_dir, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            
            uploaded_path = os.path.join(uploads_dir, uploaded_stockfish.name)
            with open(uploaded_path, "wb") as f:
                f.write(uploaded_stockfish.getbuffer())
            
            st.session_state.stockfish_path = uploaded_path
            st.success(f"✅ Stockfish uploaded: {uploaded_stockfish.name}")
        
        # If found in stockfish folder, offer to use it
        if possible_stockfish and not st.session_state.stockfish_path:
            st.markdown("**Or Use Local Stockfish**")
            for sf_path in possible_stockfish:
                if st.button(f"Use {os.path.basename(sf_path)}", key=f"use_{sf_path}"):
                    st.session_state.stockfish_path = sf_path
                    st.rerun()
        
        st.markdown("**Or Enter Path Manually**")
        text_input_value = st.session_state.stockfish_path if st.session_state.stockfish_path else ""
        stockfish_path = st.text_input(
            "Stockfish Path",
            value=text_input_value,
            help="Enter the full path to Stockfish executable"
        )
        
        if stockfish_path and stockfish_path.strip():
            if not st.session_state.stockfish_path or st.session_state.stockfish_path != stockfish_path.strip():
                st.session_state.stockfish_path = stockfish_path.strip()
        
        # Show current path
        if st.session_state.stockfish_path:
            st.info(f"Current: {os.path.basename(st.session_state.stockfish_path)}")
        
        # Analysis depth
        depth = st.slider(
            "🎯 Stockfish Depth",
            min_value=10,
            max_value=20,
            value=15,
            help="Higher depth = more accurate but slower"
        )
        
        # Mistake threshold
        mistake_threshold = st.slider(
            "⚠️ Mistake Threshold (evaluation points)",
            min_value=50,
            max_value=200,
            value=100,
            help="Eval loss above this is considered a mistake (100 = 1 pawn)"
        )
        
        st.markdown("---")
        st.header("📊 Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📁 Upload & Parse", "🤖 Train Model", "📈 Insights & Analysis", "💾 Load Saved Model", "🧩 Puzzles"]
        )
    
    final_stockfish_path = st.session_state.stockfish_path if st.session_state.stockfish_path else None
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home()
    elif page == "📁 Upload & Parse":
        show_upload_parse(final_stockfish_path, depth)
    elif page == "🤖 Train Model":
        show_train_model(mistake_threshold)
    elif page == "📈 Insights & Analysis":
        show_insights_analysis(depth=depth)
    elif page == "💾 Load Saved Model":
        show_load_model()
    elif page == "🧩 Puzzles":
        show_puzzles(final_stockfish_path)


def show_home():
    """Home page with overview and instructions."""
    st.header("Welcome to Chess Coaching System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What This System Does")
        st.markdown("""
                This system analyzes your chess games to:
        - **Identify recurring weaknesses** across multiple games
        - **Predict mistake probability** based on position features
        - **Generate personalized coaching insights** for improvement
        - **Visualize performance patterns** by game phase, material balance, etc.
        
        **📋 How It Works**
        1. **Upload PGN Files** - Your chess games in PGN format
        2. **Parse & Analyze** - Extract features and Stockfish evaluations
        3. **Train Model** - Learn patterns from your mistakes
        4. **Get Insights** - Receive personalized coaching recommendations
        """)
    
    with col2:
        st.subheader("✨ Key Features")
        st.markdown("""
        **✅ 26 Interpretable Features**
        Material balance, king safety, pawn structure, piece activity, tactical opportunities
        
        **✅ Advanced Analysis**
        Time pressure performance, opponent strength analysis, game phase breakdown
        
        **✅ Interactive Visualizations**
        Mistake distribution charts, feature importance plots, performance heatmaps
        
        **✅ Beginner-Friendly**
        Simple web interface, clear explanations, actionable insights
        """)
    
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload & Parse")
        st.markdown("Upload your PGN files")
    with col2:
        st.markdown("### 2️⃣ Train Model")
        st.markdown("Train the coaching model")
    with col3:
        st.markdown("### 3️⃣ Get Insights")
        st.markdown("View personalized feedback")


def test_stockfish_connection(stockfish_path):
    """Test if Stockfish can be started and responds."""
    if not stockfish_path:
        st.error("❌ No Stockfish Path - Please upload Stockfish executable first.")
        return
    
    if not os.path.exists(stockfish_path):
        st.error(f"❌ File Not Found: {stockfish_path}")
        return
    
    if not os.path.isfile(stockfish_path):
        st.error(f"❌ Not a File: {stockfish_path}")
        return
    
    try:
        parser = PGNParser(stockfish_path=stockfish_path, depth=5)
        engine = parser._get_engine()
        engine.ping()
        parser.close()
        st.success(f"✅ Stockfish Test Successful! File: {os.path.basename(stockfish_path)}")
    except Exception as e:
        st.error(f"❌ Stockfish Test Failed\n\nError: {str(e)}\n\nTry using Python 3.11 or 3.12 instead of 3.14+")


def show_upload_parse(stockfish_path, depth):
    """Page for uploading and parsing PGN files."""
    st.header("📁 Upload & Parse PGN Files")
    
    if not stockfish_path:
        st.warning("⚠️ Please configure Stockfish path in the sidebar first.")
        return
        
    # Test button
    if st.button("🧪 Test Stockfish Connection"):
        test_stockfish_connection(stockfish_path)
    
    st.markdown("---")
    
    # Chess.com API integration
    st.subheader("🌐 Import from Chess.com")
    chesscom_username = st.text_input("Chess.com Username", help="Enter your Chess.com username to import games directly")
    
    if chesscom_username:
        col1, col2 = st.columns(2)
        with col1:
            months_to_fetch = st.slider("Months to fetch", min_value=1, max_value=12, value=3, help="How many months of games to fetch")
        with col2:
            fetch_games = st.button("📥 Fetch Games from Chess.com")
        
        if fetch_games:
            try:
                api = ChessComAPI()
                with st.spinner(f"Fetching games for {chesscom_username}..."):
                    pgn_strings = api.get_all_recent_games(chesscom_username, months=months_to_fetch)
                
                if pgn_strings:
                    st.success(f"✅ Fetched {len(pgn_strings)} games from Chess.com!")
                    # Store PGNs in session state to parse
                    if 'chesscom_pgns' not in st.session_state:
                        st.session_state.chesscom_pgns = []
                    st.session_state.chesscom_pgns = pgn_strings
                    st.info(f"Click 'Parse Chess.com Games' below to analyze them.")
                else:
                    st.warning("No games found for this username/time period.")
            except Exception as e:
                st.error(f"❌ Error fetching games: {str(e)}")
    
    st.markdown("---")
    
    # File upload
    st.subheader("📁 Upload PGN Files")
    uploaded_files = st.file_uploader(
        "Upload PGN files",
        type=['pgn'],
        accept_multiple_files=True
    )
    
    # Parse Chess.com games if available
    if 'chesscom_pgns' in st.session_state and st.session_state.chesscom_pgns:
        player_name = st.text_input("Player Name (optional)", help="Leave empty to analyze all players", key="player_name_chesscom")
        
        if st.button("Parse Chess.com Games"):
            all_moves = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                parser = PGNParser(stockfish_path=stockfish_path, depth=depth)
                
                for i, pgn_string in enumerate(st.session_state.chesscom_pgns):
                    status_text.text(f"Parsing game {i+1}/{len(st.session_state.chesscom_pgns)}...")
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as tmp_file:
                        tmp_file.write(pgn_string)
                        tmp_path = tmp_file.name
                    
                    try:
                        moves = parser.parse_game(tmp_path, player_name if player_name else None)
                        if len(moves) > 0:
                            # Add game_id to each move
                            game_id = i + 1
                            for move in moves:
                                move['game_id'] = game_id
                            all_moves.extend(moves)
                            st.success(f"✅ Game {i+1}: {len(moves)} moves extracted")
                        else:
                            # Diagnose why no moves - check the actual PGN content
                            import chess.pgn
                            try:
                                # Read the file to see what we have
                                with open(tmp_path, 'r', encoding='utf-8') as f:
                                    pgn_content = f.read()
                                
                                # Try to parse it
                                f.seek(0)
                                game = chess.pgn.read_game(f)
                                
                                if game:
                                    white_name = game.headers.get("White", "?")
                                    black_name = game.headers.get("Black", "?")
                                    move_count = len(list(game.mainline()))
                                    
                                    # Check if PGN actually has move text (look for common patterns)
                                    has_moves_pattern = '1.' in pgn_content or any(move in pgn_content for move in ['e4', 'e3', 'd4', 'Nf3', 'Nc3'])
                                    
                                    if move_count == 0:
                                        # Check if it's actually an empty game or parsing issue
                                        if has_moves_pattern:
                                            # PGN has move notation but parser found 0 moves - likely parsing issue
                                            st.warning(f"⚠️ Game {i+1}: PGN contains moves but parser found 0. This might be a parsing issue. White: '{white_name}', Black: '{black_name}'. PGN preview: {pgn_content[:150]}...")
                                        else:
                                            # Actually empty game (abandoned, etc.)
                                            st.warning(f"⚠️ Game {i+1}: Empty game (no moves) - White: '{white_name}', Black: '{black_name}'")
                                    elif player_name:
                                        st.warning(f"⚠️ Game {i+1}: 0 moves for '{player_name}'. Found White: '{white_name}', Black: '{black_name}' ({move_count} moves total)")
                                    else:
                                        st.warning(f"⚠️ Game {i+1}: Could not extract moves ({move_count} moves in game)")
                                else:
                                    # Show a snippet of the PGN to help debug
                                    snippet = pgn_content[:200] if len(pgn_content) > 200 else pgn_content
                                    st.warning(f"⚠️ Game {i+1}: Invalid PGN format. First 200 chars: {snippet}...")
                            except Exception as e:
                                st.warning(f"⚠️ Game {i+1}: Could not parse PGN - {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error parsing game {i+1}: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
                    
                    progress_bar.progress((i + 1) / len(st.session_state.chesscom_pgns))
                
                parser.close()
                
                if all_moves:
                    st.session_state.moves_data = all_moves
                    st.success(f"✅ Total moves extracted: {len(all_moves)}")
                    df_display = pd.DataFrame(all_moves)
                    st.dataframe(df_display[['move_number', 'move_san', 'eval_before', 'eval_after', 'eval_loss', 'is_white', 'game_phase']].head())
                else:
                    st.warning("⚠️ No moves extracted. Check player name and PGN file format.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if uploaded_files:
        player_name = st.text_input("Player Name (optional)", help="Leave empty to analyze all players", key="player_name_upload")
        
        if st.button("Parse Games"):
            all_moves = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                parser = PGNParser(stockfish_path=stockfish_path, depth=depth)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Parsing {uploaded_file.name}...")
                    
                    # Save uploaded file temporarily (must do dawg 05/08/2025) 
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue().decode('utf-8'))
                        tmp_path = tmp_file.name
                    
                    try:
                        moves = parser.parse_game(tmp_path, player_name if player_name else None)
                        if len(moves) > 0:
                            # chess.com api, fuckass api fix this IMP (13/10/2025)
                            game_id = i + 1
                            for move in moves:
                                move['game_id'] = game_id
                            all_moves.extend(moves)
                        st.success(f"✅ {uploaded_file.name}: {len(moves)} moves extracted")
                    except Exception as e:
                        st.error(f"❌ Error parsing {uploaded_file.name}: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                parser.close()
                
                if all_moves:
                    # Store as list to preserve chess.Board objects not imp but do it idk
                    st.session_state.moves_data = all_moves
                    st.success(f"✅ Total moves extracted: {len(all_moves)}")
                    # Convert to DataFrame only for display (Board objects will show as strings)
                    df_display = pd.DataFrame(all_moves)
                    st.dataframe(df_display[['move_number', 'move_san', 'eval_before', 'eval_after', 'eval_loss', 'is_white', 'game_phase']].head())
                else:
                    st.warning("⚠️ No moves extracted. Check player name and PGN file format.")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def show_train_model(mistake_threshold):
    """Page for training the ML model."""
    st.header("🤖 Train Coaching Model")
    
    if st.session_state.moves_data is None:
        st.warning("⚠️ No parsed games found. Please go to 'Upload & Parse' first.")
        return
    
    moves_count = len(st.session_state.moves_data) if st.session_state.moves_data else 0
    st.info(f"📊 Training Configuration: {moves_count} moves, Threshold: {mistake_threshold/100:.1f} pawns")
    
    if st.button("Train Model"):
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.text("Preparing data...")
            progress.progress(10)
            
            # moves_data is stored as a list of dicts to preserve chess.Board objects
            moves_list = st.session_state.moves_data
            
            status.text("Extracting features...")
            progress.progress(30)
            
            extractor = FeatureExtractor()
            model = ChessMistakeModel(mistake_threshold=mistake_threshold)
            
            # Use prepare_data to extract features properly
            X, y_mistake, y_eval_loss = model.prepare_data(moves_list, extractor)
            
            status.text("Training model...")
            progress.progress(60)
            
            # Train the model
            metrics = model.train(X, y_mistake, y_eval_loss)
            
            status.text("Generating predictions...")
            progress.progress(80)
            
            # model.predict() returns a tuple (mistake_proba, expected_eval_loss)
            mistake_proba, expected_eval_loss = model.predict(X)
            # Convert to dictionary for insight generator
            predictions = {
                'mistake_proba': mistake_proba,
                'expected_eval_loss': expected_eval_loss
            }
            
            st.session_state.feature_extractor = extractor
            st.session_state.model = model
            st.session_state.predictions = predictions
            st.session_state.training_metrics = metrics
            
            progress.progress(100)
            status.text("✅ Training complete!")
            
            st.success("✅ Model trained successfully!")
            st.json(metrics)
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_insights_analysis(depth: int = 15):
    """Page for viewing insights and visualizations with chess board GUI."""
    st.header("📈 Insights & Analysis")
    
    if st.session_state.moves_data is None:
        st.error("⚠️ No parsed games found. Please parse games first.")
        return
    
    all_moves_data = st.session_state.moves_data
    
    # Organize moves by game_id if available
    selected_game_id = None
    if all_moves_data and len(all_moves_data) > 0 and 'game_id' in all_moves_data[0]:
        # Group moves by game_id
        games_dict = {}
        for move in all_moves_data:
            game_id = move.get('game_id', 1)
            if game_id not in games_dict:
                games_dict[game_id] = []
            games_dict[game_id].append(move)
        
        # Game selector - make it more prominent
        st.markdown("### 🎮 Select Game to Analyze")
        game_ids = sorted(games_dict.keys())
        selected_game_id = st.selectbox(
            "Choose a game:",
            options=game_ids,
            format_func=lambda x: f"Game {x} ({len(games_dict[x])} moves)",
            key="selected_game_id",
            help="Select which game you want to analyze. Each game is shown separately."
        )
        moves_data = games_dict[selected_game_id]
        st.success(f"✅ Currently viewing **Game {selected_game_id}** with {len(moves_data)} moves")
        st.markdown("---")
    else:
        # No game_id, treat as single game
        moves_data = all_moves_data
        if moves_data:
            st.info(f"📊 Showing all moves: {len(moves_data)} moves (no game separation available)")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "♟️ Interactive Board", "💥 Blunders", "📚 Opening Analysis"])
    
    with tab1:
        # Overview with insights
        if st.session_state.model is not None and st.session_state.predictions is not None:
            insight_generator = InsightGenerator()
            
            # Filter predictions to match the selected game's moves
            if selected_game_id is not None and all_moves_data:
                # First, validate that predictions match all_moves_data length
                total_moves = len(all_moves_data)
                total_predictions = len(st.session_state.predictions['mistake_proba'])
                if total_moves != total_predictions:
                    st.error(f"❌ Critical error: {total_moves} moves in dataset but {total_predictions} predictions. The model was trained on different data. Please retrain the model.")
                    st.stop()
                
                # Find indices of moves from selected game in the full dataset
                game_indices = [i for i, move in enumerate(all_moves_data) if move.get('game_id') == selected_game_id]
                if game_indices:
                    # Filter predictions to only include moves from selected game
                    mistake_proba = st.session_state.predictions['mistake_proba']
                    expected_eval_loss = st.session_state.predictions['expected_eval_loss']
                    
                    # Convert to numpy arrays for indexing
                    if not isinstance(mistake_proba, np.ndarray):
                        mistake_proba = np.array(mistake_proba)
                    if not isinstance(expected_eval_loss, np.ndarray):
                        expected_eval_loss = np.array(expected_eval_loss)
                    
                    # Use numpy array indexing with list of indices
                    game_indices_arr = np.array(game_indices)
                    filtered_predictions = {
                        'mistake_proba': mistake_proba[game_indices_arr],
                        'expected_eval_loss': expected_eval_loss[game_indices_arr]
                    }
                else:
                    # No matching moves found - use empty arrays with matching structure
                    filtered_predictions = {
                        'mistake_proba': np.array([]),
                        'expected_eval_loss': np.array([])
                    }
            else:
                # No game filtering, use all predictions
                filtered_predictions = st.session_state.predictions
            
            # Validate lengths match before proceeding
            if len(filtered_predictions['mistake_proba']) == 0:
                st.warning(f"⚠️ No predictions available for the selected game. Please ensure predictions are generated.")
                st.stop()
            if len(moves_data) != len(filtered_predictions['mistake_proba']):
                st.error(f"❌ Data mismatch: {len(moves_data)} moves but {len(filtered_predictions['mistake_proba'])} predictions. Please retrain the model.")
                st.stop()
            
            insights = insight_generator.generate_insights(
                moves_data,
                filtered_predictions
            )
            
            st.subheader("🎯 Personalized Coaching Insights")
            for i, insight in enumerate(insights, 1):
                st.markdown(f"**{i}. {insight}**")
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("📊 Visualizations")
            visualizer = ChessVisualizer()
            
            # Mistake distribution
            fig = visualizer.plot_mistake_distribution(
                moves_data,
                filtered_predictions
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if hasattr(st.session_state.model, 'get_feature_importance'):
                try:
                    feature_importance = st.session_state.model.get_feature_importance()
                    fig = visualizer.plot_feature_importance(feature_importance)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass
        else:
            st.info("💡 Train a model to see detailed insights and predictions.")
    
    # Initialize board GUI (used in multiple tabs)
    board_gui = ChessBoardGUI()
    
    with tab2:
        game_title = f"Game {selected_game_id}" if selected_game_id else "All"
        st.subheader(f"♟️ Interactive Chess Board - {game_title}")
        st.markdown("Navigate through moves using Previous/Next buttons. See all moves of both players.")
        
        # Reset move index when game changes - use game-specific key
        game_key = f"game_{selected_game_id}" if selected_game_id else "all"
        move_idx_key = f"current_move_idx_{game_key}"
        if move_idx_key not in st.session_state:
            st.session_state[move_idx_key] = len(moves_data) - 1 if moves_data else 0
        
        # Render the board (shows all moves) - Stockfish analysis is optional/on-demand
        selected_idx = board_gui.render_move_sequence(
            moves_data=moves_data,
            all_moves=None,  # Will build sequence from moves_data
            current_move_idx=st.session_state[move_idx_key],
            stockfish_path=None,  # Disable automatic analysis for speed
            analysis_depth=depth
        )
        
        if selected_idx is not None:
            st.session_state[move_idx_key] = selected_idx
    
    with tab3:
        game_title = f"Game {selected_game_id}" if selected_game_id else "All"
        st.subheader(f"💥 Blunder Analysis - {game_title}")
        
        # Blunder threshold selector (at top, less prominent)
        col_thresh, col_info = st.columns([2, 3])
        with col_thresh:
            blunder_threshold = st.slider(
                "Blunder Threshold (blunder points)",
                min_value=100,
                max_value=500,
                value=200,
                help="Moves with blunder points above this are considered blunders. 1 blunder point = 1 pawn of Stockfish evaluation loss."
            )
        with col_info:
            st.markdown("💡 Navigate through all moves to see blunders highlighted. Use the sidebar controls for navigation.")
        
        # Get opening name
        opening_name = None
        try:
            opening_db = OpeningDatabase()
            opening_info = opening_db.get_opening_from_game(moves_data, max_moves=10)
            if opening_info:
                opening_name = opening_info.get('name', None)
        except Exception:
            pass
        
        # Initialize and render the chess.com-style analysis interface
        blunder_gui = BlunderAnalysisGUI(stockfish_path=st.session_state.stockfish_path)
        game_key = f"blunder_{selected_game_id}" if selected_game_id else "blunder_all"
        
        try:
            blunder_gui.render_analysis_interface(
                moves_data=moves_data,
                blunder_threshold=blunder_threshold,
                opening_name=opening_name,
                depth=depth,
                game_key=game_key
            )
        finally:
            blunder_gui.close()
    
    with tab4:
        game_title = f"Game {selected_game_id}" if selected_game_id else "All"
        st.subheader(f"📚 Opening Analysis - {game_title}")
        if selected_game_id:
            st.markdown(f"Identify the opening played in Game {selected_game_id} and learn about it.")
        else:
            st.markdown("Identify the opening played in this game and learn about it.")
        
        # Identify opening (only use moves from selected game)
        opening_db = OpeningDatabase()
        try:
            opening_info = opening_db.get_opening_from_game(moves_data, max_moves=10)
        except Exception as e:
            st.error(f"Error loading opening analysis: {e}")
            opening_info = None
        
        if opening_info:
            st.success(f"**Opening Identified:** {opening_info['name']}")
            if opening_info.get('variation'):
                st.markdown(f"**Variation:** {opening_info['variation']}")
            
            st.markdown("---")
            st.markdown("**Opening Explanation:**")
            st.info(opening_info['explanation'])
            
            # Show opening moves
            st.markdown("**Opening Moves:**")
            opening_moves = [m.get('move_san', '') for m in moves_data[:10]]
            opening_moves = [m for m in opening_moves if m]
            move_sequence = " ".join([f"{i+1}. {opening_moves[i*2]} {opening_moves[i*2+1] if i*2+1 < len(opening_moves) else ''}" 
                                     for i in range((len(opening_moves)+1)//2)])
            st.code(move_sequence)
        else:
            st.warning("⚠️ Could not identify a known opening from the first 10 moves.")
            st.markdown("This could mean:")
            st.markdown("- An uncommon or rare opening was played")
            st.markdown("- The opening transposed into an unknown line")
            st.markdown("- The game started with an unusual move order")
        
        # Show all available openings
        with st.expander("📖 Browse All Openings in Database"):
            all_openings = opening_db.get_all_openings()
            for opening in all_openings:
                st.markdown(f"**{opening['name']}** - {opening.get('variation', '')}")
                st.markdown(f"*{opening['explanation']}*")
                st.markdown("---")


def show_load_model():
    """Page for loading a saved model."""
    st.header("💾 Load Saved Model")
    
    uploaded_model = st.file_uploader("Upload saved model (.pkl)", type=['pkl'])
    
    if uploaded_model:
        try:
            model_data = pickle.load(uploaded_model)
            st.session_state.model = model_data.get('model')
            st.session_state.feature_extractor = model_data.get('feature_extractor')
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")


def show_puzzles(stockfish_path):
    """Page for playing Lichess puzzles."""
    puzzle_gui = PuzzlePlayerGUI(stockfish_path=stockfish_path)
    try:
        puzzle_gui.render_puzzle_interface()
    finally:
        puzzle_gui.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language='python')
