# System Design: Human Mistake Modeling for Chess Coaching

## Overview

This system models **human mistakes in chess**, not chess strength. It's designed to identify recurring weaknesses and provide personalized coaching insights for ~1200-rated players.

## Core Philosophy

**What this is:**
- A mistake prediction system
- A pattern discovery tool
- An explainable ML system for coaching

**What this is NOT:**
- A chess engine (we use Stockfish only for evaluation)
- Reinforcement learning
- A black-box deep learning model
- A system that suggests best moves

## ML Pipeline

### 1. Data Collection (PGN Parser)

**Input:** PGN game files

**Process:**
- Parse PGN files using `python-chess`
- For each move:
  - Get Stockfish evaluation **before** the move
  - Get Stockfish evaluation **after** the move
  - Compute **eval loss** = eval_before - eval_after (from player's perspective)

**Output:** List of moves with:
- Board states (before/after)
- Stockfish evaluations
- Eval loss (ground truth label)
- Game context (phase, move number, etc.)

**Key Design Decision:** Stockfish is used **only** as a ground-truth evaluator, not as a decision maker. We're learning from human mistakes, not trying to play like Stockfish.

### 2. Feature Extraction

**Input:** Chess board positions

**Output:** 21 interpretable features

**Feature Categories:**

1. **Material (5 features)**
   - Material balance (normalized)
   - Pawn difference
   - Minor piece difference (knights + bishops)
   - Major piece difference (rooks)
   - Queen difference

2. **King Safety (3 features)**
   - King rank (how advanced)
   - Pawn shield strength
   - King activity

3. **Pawn Structure (4 features)**
   - Doubled pawns ratio
   - Isolated pawns ratio
   - Passed pawns ratio
   - Pawn advancement

4. **Piece Activity (4 features)**
   - Average piece mobility
   - Center control
   - Piece coordination
   - Development ratio

5. **Game Phase (1 feature)**
   - Phase (0 = opening, 1 = endgame)

6. **Board Control (2 features)**
   - Center control
   - Overall mobility

7. **Tactical (2 features)**
   - Hanging pieces
   - Tactical opportunities

**Why These Features?**
- **Interpretable:** Each feature has chess meaning
- **Domain Knowledge:** Based on chess principles, not raw board states
- **Generalizable:** Work across different positions and games
- **Beginner-Friendly:** Features align with chess concepts players learn

### 3. Label Construction

**Classification Label (Mistake/Not Mistake):**
- If `eval_loss > threshold` (default: 100 centipawns) → **Mistake (1)**
- Otherwise → **Not Mistake (0)**

**Regression Label (Eval Loss):**
- Direct use of `eval_loss` value in centipawns

**Why This Works:**
- Stockfish provides objective ground truth
- Eval loss captures move quality from player's perspective
- Threshold-based classification is interpretable (100 cp ≈ losing a pawn)

### 4. Model Training

**Model Choice: Random Forest**

**Why Random Forest?**
1. **Non-linear patterns:** Chess mistakes have complex patterns
2. **Interpretability:** Feature importance shows what matters
3. **Robustness:** Less prone to overfitting than deep networks
4. **Speed:** Fast training and inference
5. **Mixed features:** Handles different feature types well

**Two Models:**
1. **Classifier:** Predicts mistake probability (0-1)
2. **Regressor:** Predicts expected eval loss (centipawns)

**Hyperparameters:**
- `n_estimators=100`: Good balance of accuracy and speed
- `max_depth=15`: Prevents overfitting
- `min_samples_split=10`: Regularization
- `min_samples_leaf=5`: Regularization

**Training Strategy:**
- **Game-level splits:** Split by game, not by move (avoids data leakage)
- **Cross-validation:** Train on multiple games, validate on held-out games
- **Feature normalization:** Some features already normalized, others handled by tree splits

### 5. Avoiding Overfitting

**Problem:** Model might memorize Stockfish evaluations rather than learn human mistake patterns.

**Solutions:**

1. **Feature Engineering**
   - Use domain knowledge (material, king safety) instead of raw board
   - Features are abstractions, not exact positions

2. **Regularization**
   - Limit tree depth
   - Require minimum samples per split/leaf
   - Limit number of trees

3. **Game-Level Splits**
   - Don't split moves randomly
   - Keep entire games together (train/test split by game)

4. **Threshold Tuning**
   - Don't overfit to single mistake threshold
   - Model learns patterns, not just threshold

5. **Feature Diversity**
   - 21 features cover different aspects
   - Reduces reliance on single features

### 6. Insight Generation

**Process:**
1. Generate predictions for all moves
2. Aggregate by game phase, material balance, etc.
3. Identify patterns:
   - High mistake rate in specific phases
   - Performance when ahead/behind
   - Recurring mistake types
4. Generate human-readable insights

**Example Insights:**
- "You consistently lose material in endgames (avg eval loss: 150 cp)"
- "King safety is a weakness in middlegame positions"
- "You perform poorly when behind in material"

**Why This Works:**
- Aggregates across many games (generalization)
- Focuses on recurring patterns (not single positions)
- Uses model predictions + actual outcomes
- Provides actionable feedback

## Data Flow

```
PGN Files
    ↓
PGN Parser + Stockfish
    ↓
Moves with evaluations
    ↓
Feature Extractor
    ↓
Feature vectors (X) + Labels (y)
    ↓
Model Training
    ↓
Trained Models (Classifier + Regressor)
    ↓
Predictions on new games
    ↓
Insight Generator
    ↓
Coaching Insights
```

## Model Justification

### Why Not Deep Learning?
- **Interpretability:** Random Forest provides feature importance
- **Data efficiency:** Works well with limited data (hundreds of games)
- **Simplicity:** Easier to understand and debug
- **Overfitting:** Less prone to overfitting with small datasets

### Why Not Reinforcement Learning?
- **Different problem:** We're modeling mistakes, not learning to play
- **Supervised learning:** We have labels (Stockfish evaluations)
- **Coaching focus:** We want insights, not optimal play

### Why Not Gradient Boosting (XGBoost)?
- **Could work:** XGBoost would likely perform better
- **Trade-off:** Slightly less interpretable, more complex
- **Random Forest is sufficient:** Good performance with better interpretability

## Limitations & Future Improvements

**Current Limitations:**
1. **Feature extraction:** Some features are simplified (e.g., king activity)
2. **Game-level tracking:** Currently assumes one game per PGN file
3. **Time control:** Doesn't account for time pressure
4. **Opponent strength:** Doesn't consider opponent rating

**Future Improvements:**
1. **More sophisticated features:** Better king safety, piece coordination
2. **Temporal features:** Track position changes over time
3. **Context features:** Time remaining, move number, etc.
4. **Ensemble methods:** Combine multiple models
5. **Deep features:** Use pre-trained chess embeddings (if available)

## Usage Example

```python
# 1. Parse games
parser = PGNParser(stockfish_path="stockfish", depth=15)
moves_data = parser.parse_games_directory("games/", player_name="PlayerName")

# 2. Extract features
feature_extractor = FeatureExtractor()
model = ChessMistakeModel(mistake_threshold=100.0)
X, y_mistake, y_eval_loss = model.prepare_data(moves_data, feature_extractor)

# 3. Train
metrics = model.train(X, y_mistake, y_eval_loss, test_size=0.2)

# 4. Generate insights
mistake_proba, expected_loss = model.predict(X)
insights = InsightGenerator().generate_insights(moves_data, {
    'mistake_proba': mistake_proba,
    'expected_eval_loss': expected_loss
})
```

## Summary

This system:
- ✅ Models human mistakes (not chess strength)
- ✅ Uses supervised learning with Stockfish labels
- ✅ Extracts interpretable features
- ✅ Trains Random Forest models
- ✅ Generates coaching insights
- ✅ Avoids overfitting through regularization and game-level splits
- ✅ Provides explainable, actionable feedback

The approach is **practical, interpretable, and focused on coaching** rather than playing chess optimally.

