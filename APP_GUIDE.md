# Chess Coaching System - Web App Guide

## 🚀 Getting Started

### Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Stockfish:**
   - **Windows:** Download from [stockfishchess.org](https://stockfishchess.org/download/)
   - **Linux:** `sudo apt-get install stockfish`
   - **Mac:** `brew install stockfish`

3. **Launch the app:**
```bash
streamlit run app.py
```

Or use the launcher:
```bash
python run_app.py
```

4. **Open your browser** to the URL shown (usually `http://localhost:8501`)

## 📱 App Interface

### Navigation

The app has 5 main pages accessible from the sidebar:

1. **🏠 Home** - Overview and quick start guide
2. **📁 Upload & Parse** - Upload PGN files and parse with Stockfish
3. **🤖 Train Model** - Train the ML coaching model
4. **📈 Insights & Analysis** - View coaching insights and visualizations
5. **💾 Load Saved Model** - Load a previously trained model

### Sidebar Configuration

- **Stockfish Path:** Path to Stockfish executable (or 'stockfish' if in PATH)
- **Stockfish Depth:** Analysis depth (10-20, higher = more accurate but slower)
- **Mistake Threshold:** Eval loss threshold in centipawns (50-200)

## 📋 Step-by-Step Workflow

### Step 1: Upload & Parse Games

1. Go to **📁 Upload & Parse** page
2. Click "Browse files" and select your PGN files
3. (Optional) Enter your player name to analyze only your moves
4. Click **🚀 Parse Games**
5. Wait for parsing to complete (this may take a while depending on number of games and Stockfish depth)

**What happens:**
- PGN files are parsed
- Each move is analyzed by Stockfish
- Features are extracted from positions
- Summary statistics are displayed

### Step 2: Train Model

1. Go to **🤖 Train Model** page
2. Review the number of moves to be trained on
3. Click **🚀 Train Model**
4. Wait for training to complete

**What happens:**
- Features are extracted from all positions
- Random Forest models are trained (classifier + regressor)
- Model performance metrics are displayed
- Feature importance is shown

**Optional:** Save the model for later use

### Step 3: View Insights & Analysis

1. Go to **📈 Insights & Analysis** page
2. Review coaching insights at the top
3. Explore different visualization tabs:
   - **Phase Analysis:** Performance by game phase
   - **Material Analysis:** Performance when ahead/behind
   - **Time Pressure:** Performance under time pressure (if available)
   - **Feature Importance:** Which features matter most
   - **Mistake Heatmap:** Mistakes by phase and move number
   - **Prediction Accuracy:** Model performance metrics

## 🎯 Understanding the Insights

### Coaching Insights

The system generates insights like:

- **"Overall Performance: 28.5% of moves are predicted mistakes"**
  - Your overall mistake rate

- **"Endgame Weakness: 35.2% mistake rate"**
  - You make more mistakes in endgames

- **"When Behind: 42.1% mistake rate when losing"**
  - You struggle when behind in material

- **"Recurring Pattern: Most mistakes occur in the middlegame"**
  - Common mistake patterns

### Visualizations

- **Mistake Rate by Phase:** Bar chart showing mistake rate in opening/middlegame/endgame
- **Eval Loss Distribution:** Box plots showing eval loss distribution
- **Material Balance Analysis:** Performance when ahead/equal/behind
- **Feature Importance:** Which chess concepts matter most for your mistakes
- **Mistake Heatmap:** Visual map of where mistakes occur
- **Prediction Accuracy:** Confusion matrix showing model performance

## 💡 Tips

1. **More Games = Better Model**
   - Train on 50+ games for best results
   - More diverse games = better generalization

2. **Adjust Mistake Threshold**
   - Lower threshold (50-75 cp) = more sensitive
   - Higher threshold (150-200 cp) = only major mistakes
   - Default 100 cp ≈ losing a pawn

3. **Stockfish Depth**
   - Depth 10-12: Fast, good for testing
   - Depth 15: Balanced (recommended)
   - Depth 18-20: Very accurate but slow

4. **Player Name**
   - Leave empty to analyze both players
   - Enter exact name (case-sensitive) to analyze one player

5. **Save Your Models**
   - Save trained models for later use
   - Load saved models to analyze new games

## 🔧 Troubleshooting

### "Failed to start Stockfish"
- Make sure Stockfish is installed
- Check the Stockfish path in sidebar
- Try full path: `C:/path/to/stockfish.exe` (Windows) or `/usr/bin/stockfish` (Linux)

### "No moves extracted"
- Check PGN file format
- Verify player name matches exactly
- Make sure games have moves

### App is slow
- Reduce Stockfish depth
- Process fewer games at once
- Use faster hardware

### Model overfitting
- Train on more games (100+ recommended)
- Use game-level train/test splits (already implemented)
- Check feature importance for unusual patterns

## 📊 Advanced Features

### Enhanced Features (26 total)

The system now includes:
- **Basic features (21):** Material, king safety, pawn structure, piece activity, etc.
- **Advanced features (5):** Piece-square scores, bishop pair, rook on open file, knight outposts, piece coordination

### Time Pressure Analysis

If PGN files include time information in comments, the system analyzes:
- Performance under time pressure
- Mistake rate with different time remaining
- Time management insights

### Opponent Strength Analysis

If PGN files include ratings, the system analyzes:
- Performance vs different opponent strengths
- Rating-based mistake patterns
- Matchup insights

## 🎓 Next Steps

1. **Collect more games** for better model accuracy
2. **Experiment with thresholds** to find what works for you
3. **Review feature importance** to understand what matters
4. **Focus on weak areas** identified by insights
5. **Track improvement** by retraining on new games

## 📚 Learn More

- See `DESIGN.md` for system architecture
- See `QUICKSTART.md` for command-line usage
- See `README.md` for overview

