"""
Generate Coaching Insights

Loads trained model and generates insights from PGN games.
"""

import argparse
import os
from pgn_parser import PGNParser
from feature_extractor import FeatureExtractor
from model_trainer import ChessMistakeModel
from insight_generator import InsightGenerator
import pickle


def main():
    parser = argparse.ArgumentParser(description='Generate coaching insights')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pkl file)')
    parser.add_argument('--games_dir', type=str, required=True,
                       help='Directory containing PGN files')
    parser.add_argument('--player_name', type=str, default=None,
                       help='Name of player to analyze')
    parser.add_argument('--stockfish_path', type=str, default=None,
                       help='Path to Stockfish executable')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for insights (if None, prints to console)')
    parser.add_argument('--depth', type=int, default=15,
                       help='Stockfish analysis depth')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING COACHING INSIGHTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games directory: {args.games_dir}")
    print(f"Player: {args.player_name or 'Both players'}")
    print("=" * 60)
    print()
    
    # Load model
    print("Loading trained model...")
    model = ChessMistakeModel.load(args.model)
    print("Model loaded successfully\n")
    
    # Load feature extractor (from same directory as model)
    model_dir = os.path.dirname(args.model)
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pkl')
    
    if os.path.exists(feature_extractor_path):
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
    else:
        print("Feature extractor not found, using new instance")
        feature_extractor = FeatureExtractor()
    
    # Parse games
    print("Parsing games and extracting features...")
    parser_obj = PGNParser(stockfish_path=args.stockfish_path, depth=args.depth)
    
    try:
        moves_data = parser_obj.parse_games_directory(args.games_dir, args.player_name)
        print(f"Extracted {len(moves_data)} moves\n")
        
        if len(moves_data) == 0:
            print("ERROR: No moves extracted.")
            return
        
    except Exception as e:
        print(f"ERROR parsing games: {e}")
        return
    finally:
        parser_obj.close()
    
    # Extract features
    print("Extracting features...")
    X, _, _ = model.prepare_data(moves_data, feature_extractor)
    print(f"Extracted features for {X.shape[0]} positions\n")
    
    # Generate predictions
    print("Generating predictions...")
    mistake_proba, expected_eval_loss = model.predict(X)
    
    predictions = {
        'mistake_proba': mistake_proba,
        'expected_eval_loss': expected_eval_loss
    }
    
    # Generate insights
    print("Generating insights...\n")
    insight_generator = InsightGenerator()
    report = insight_generator.generate_detailed_report(moves_data, predictions)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Insights saved to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()

