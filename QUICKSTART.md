# Quick Start Guide

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Stockfish:**
   - **Windows:** Download from [stockfishchess.org](https://stockfishchess.org/download/) or use `winget install Stockfish.Stockfish`
   - **Linux:** `sudo apt-get install stockfish` or `sudo yum install stockfish`
   - **Mac:** `brew install stockfish`

3. **Verify Stockfish:**
```bash
stockfish
# Type "quit" to exit
```

## Basic Usage

### Step 1: Prepare Your Games

Place your PGN files in a directory (e.g., `games/`). Each file should contain one or more games.

### Step 2: Train the Model

```bash
python train_coaching_model.py --games_dir games/ --output_dir models/
```

**Options:**
- `--player_name "YourName"` - Analyze specific player only
- `--mistake_threshold 100.0` - Eval loss threshold for mistakes (centipawns)
- `--depth 15` - Stockfish analysis depth (higher = more accurate, slower)
- `--stockfish_path "/path/to/stockfish"` - If Stockfish not in PATH

**Example:**
```bash
python train_coaching_model.py \
    --games_dir games/ \
    --output_dir models/ \
    --player_name "John Doe" \
    --mistake_threshold 100.0 \
    --depth 15
```

### Step 3: Generate Insights

```bash
python generate_insights.py \
    --model models/coaching_model.pkl \
    --games_dir games/ \
    --player_name "John Doe" \
    --output insights.txt
```

## Example Output

```
============================================================
CHESS COACHING INSIGHTS REPORT
============================================================

1. Overall Performance: 28.5% of moves are predicted mistakes. Average eval loss: 85.3 centipawns.

2. Endgame Weakness: 35.2% mistake rate, avg eval loss 120.5 cp. Consider focusing on endgame study.

3. When Behind: 42.1% mistake rate when losing, avg loss 145.8 cp. Focus on defensive techniques.

4. Recurring Pattern: Most mistakes occur in the middlegame. Consider targeted study in this phase.

5. Consistency: High variance in move quality. Focus on maintaining consistent play.

============================================================
```

## Python API Usage

```python
from pgn_parser import PGNParser
from feature_extractor import FeatureExtractor
from model_trainer import ChessMistakeModel
from insight_generator import InsightGenerator

# 1. Parse games
parser = PGNParser(stockfish_path="stockfish", depth=15)
moves_data = parser.parse_games_directory("games/", player_name="YourName")
parser.close()

# 2. Extract features and prepare data
feature_extractor = FeatureExtractor()
model = ChessMistakeModel(mistake_threshold=100.0)
X, y_mistake, y_eval_loss = model.prepare_data(moves_data, feature_extractor)

# 3. Train
metrics = model.train(X, y_mistake, y_eval_loss, test_size=0.2)
print(f"Accuracy: {metrics['mistake_accuracy']:.3f}")

# 4. Generate insights
mistake_proba, expected_loss = model.predict(X)
insights = InsightGenerator().generate_insights(moves_data, {
    'mistake_proba': mistake_proba,
    'expected_eval_loss': expected_loss
})

for insight in insights:
    print(insight)
```

## Troubleshooting

### Stockfish Not Found

**Error:** `Failed to start Stockfish`

**Solution:**
1. Make sure Stockfish is installed
2. Specify path: `--stockfish_path "C:/path/to/stockfish.exe"` (Windows) or `--stockfish_path "/usr/bin/stockfish"` (Linux)

### No Moves Extracted

**Error:** `No moves extracted`

**Solution:**
1. Check PGN files are valid
2. Verify player name matches exactly (case-sensitive)
3. Make sure games directory path is correct

### Slow Processing

**Solutions:**
1. Reduce Stockfish depth: `--depth 12` (faster, less accurate)
2. Process fewer games
3. Use faster hardware or cloud computing

### Model Overfitting

**Symptoms:** High training accuracy, low test accuracy

**Solutions:**
1. Use more games (100+ games recommended)
2. Increase regularization (modify `model_trainer.py`)
3. Use game-level train/test splits (already implemented)

## Next Steps

1. **Collect more games:** More data = better model
2. **Tune hyperparameters:** Adjust mistake threshold, tree depth, etc.
3. **Add features:** Extend `feature_extractor.py` with more features
4. **Customize insights:** Modify `insight_generator.py` for specific analysis

## File Structure

```
chess_coaching_system/
├── pgn_parser.py          # PGN parsing + Stockfish integration
├── feature_extractor.py   # Feature extraction from positions
├── model_trainer.py       # ML model training
├── insight_generator.py   # Coaching insight generation
├── train_coaching_model.py # Training script
├── generate_insights.py   # Insight generation script
├── example_usage.py       # Example workflow
├── requirements.txt       # Python dependencies
├── README.md             # Main documentation
├── DESIGN.md             # System design explanation
└── QUICKSTART.md         # This file
```

## Questions?

See `DESIGN.md` for detailed explanations of:
- Why Random Forest?
- How labels are constructed
- How to avoid overfitting
- Feature engineering rationale

