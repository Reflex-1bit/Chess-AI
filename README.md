# Chess Coaching System: Human Mistake Modeling

A machine learning system that learns from a player's chess games to identify recurring weaknesses and provide personalized coaching insights.

## Overview

This system analyzes PGN games to:
- Extract per-move features (material, king safety, pawn structure, etc.)
- Use Stockfish to compute evaluation losses (ground truth)
- Train a model to predict mistake probability and expected eval loss
- Aggregate patterns across games to identify recurring weaknesses
- Generate human-readable coaching insights

## Approach

**Not a chess engine** - This models human mistakes, not chess strength.

**Supervised Learning** - Uses Stockfish eval loss as labels to learn when players make mistakes.

**Feature-Based** - Extracts interpretable features (material balance, king safety, etc.) rather than raw board states.

**Pattern Discovery** - Aggregates predictions across many games to find recurring weaknesses.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** You'll need Stockfish installed on your system. Download from [stockfishchess.org](https://stockfishchess.org/download/) or install via package manager.

## Quick Start

### Option 1: Web App (Recommended) 🚀

**Windows/Mac/Linux Users:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**What happens:**
- The app will start and open in your browser automatically
- If not, go to: http://localhost:8501
- Follow the app interface:
  - Upload PGN files or fetch from Chess.com
  - Parse games with Stockfish analysis
  - Train model on your games
  - View insights, interactive board, blunders, and puzzles

### Option 2: Command Line

1. Place your PGN files in a directory (e.g., `games/`)
2. Run the training pipeline:

```bash
python train_coaching_model.py --games_dir games/ --output_dir models/
```

3. Generate insights for a player:

```bash
python generate_insights.py --model models/coaching_model.pkl --games_dir games/
```

## Project Structure

### Core Application
- `app.py` - Main Streamlit web application
- `pgn_parser.py` - Parses PGN files and extracts moves with Stockfish evaluations
- `feature_extractor.py` - Converts chess positions into feature vectors
- `model_trainer.py` - Trains ML model on extracted features and labels
- `insight_generator.py` - Aggregates predictions and generates coaching insights
- `visualizations.py` - Creates interactive charts and graphs

### GUI Components
- `chess_board_gui.py` - Interactive chess board visualization
- `chess_board_visualizer.py` - HTML/CSS board renderer
- `blunder_analyzer.py` - Identifies and explains blunders
- `blunder_gui.py` - Blunder analysis interface
- `puzzle_player_gui.py` - Lichess puzzle player interface
- `puzzle_recommender.py` - AI-powered puzzle recommendations
- `puzzle_themes.py` - Lichess puzzle theme mappings

### API Integrations
- `chesscom_api.py` - Chess.com API integration
- `lichess_puzzles.py` - Lichess puzzle API integration
- `opening_database.py` - Opening database and analysis

### Command Line Tools
- `train_coaching_model.py` - Training pipeline (CLI)
- `generate_insights.py` - Insight generation (CLI)

## Model Choice: Random Forest

**Why Random Forest?**
- Handles non-linear patterns in chess mistakes
- Provides feature importance (interpretability)
- Robust to overfitting with proper hyperparameters
- Fast training and inference
- Works well with mixed feature types

**Alternative:** Gradient Boosting (XGBoost) for better accuracy, but slightly less interpretable.

## Label Construction

For each move:
1. Get Stockfish evaluation before the move
2. Get Stockfish evaluation after the move
3. Compute **eval loss** = eval_before - eval_after (from player's perspective)
4. Label as **mistake** if eval_loss > threshold (e.g., 100 centipawns)

This creates a supervised learning problem: predict mistake probability from board features.

## Avoiding Overfitting

1. **Feature engineering** - Use domain knowledge (material, king safety) rather than raw board
2. **Cross-validation** - Train on multiple games, validate on held-out games
3. **Regularization** - Limit tree depth in Random Forest
4. **Game-level splits** - Don't split moves randomly; split by game to avoid data leakage
5. **Threshold tuning** - Don't overfit to a single mistake threshold

## Features

- **Interactive Game Analysis**: Visualize games move-by-move with Stockfish evaluation
- **Blunder Detection**: Automatic identification of mistakes with detailed explanations
- **AI-Powered Insights**: Machine learning identifies recurring weaknesses and patterns
- **Puzzle Recommendations**: Personalized puzzle suggestions based on your weaknesses
- **Chess.com Integration**: Directly fetch and analyze your Chess.com games
- **Comprehensive Visualizations**: Charts showing performance by phase, material, and more
- **Lichess Puzzle Player**: Practice puzzles with interactive board

## Output Format

The system generates insights like:
- "You consistently lose material in endgames (avg blunder points: 1.5)"
- "King safety is a weakness in middlegame positions"
- "You perform poorly when behind in material"
- Personalized puzzle theme recommendations (e.g., "Focus on forks and pins")

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here - MIT, Apache, etc.]

## Acknowledgments

- Stockfish chess engine for position evaluation
- Lichess for puzzle API
- Chess.com for game data API

