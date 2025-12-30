"""
Feature Extraction for Chess Positions

Converts chess positions into interpretable feature vectors for ML models.
Features include material balance, king safety, pawn structure, piece activity, etc.
"""

import chess
import numpy as np
from typing import List
from collections import defaultdict


class FeatureExtractor:
    """Extracts interpretable features from chess positions."""
    
    # Piece values in centipawns (standard)
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  # Not used in material calculation
    }
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_features(self, board: chess.Board, is_white: bool) -> np.ndarray:
        """
        Extract feature vector from a chess position.
        
        Args:
            board: Chess board position
            is_white: Whether features are from white's perspective
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Material features (5 features)
        material_features = self._extract_material_features(board, is_white)
        features.extend(material_features)
        
        # King safety features (3 features)
        king_safety = self._extract_king_safety(board, is_white)
        features.extend(king_safety)
        
        # Pawn structure features (4 features)
        pawn_structure = self._extract_pawn_structure(board, is_white)
        features.extend(pawn_structure)
        
        # Piece activity features (4 features)
        piece_activity = self._extract_piece_activity(board, is_white)
        features.extend(piece_activity)
        
        # Game phase feature (1 feature)
        game_phase = self._get_game_phase_numeric(board)
        features.append(game_phase)
        
        # Control features (2 features)
        control = self._extract_control_features(board, is_white)
        features.extend(control)
        
        # Tactical features (2 features)
        tactical = self._extract_tactical_features(board, is_white)
        features.extend(tactical)
        
        # Advanced features (5 features)
        advanced = self._extract_advanced_features(board, is_white)
        features.extend(advanced)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_material_features(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract material balance features.
        
        Returns:
            [material_balance, pawn_diff, minor_piece_diff, major_piece_diff, queen_diff]
        """
        white_material = 0
        black_material = 0
        white_pawns = 0
        black_pawns = 0
        white_minors = 0  # Knights + Bishops
        black_minors = 0
        white_majors = 0  # Rooks
        black_majors = 0
        white_queens = 0
        black_queens = 0
        
        for square, piece in board.piece_map().items():
            value = self.PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
                if piece.piece_type == chess.PAWN:
                    white_pawns += 1
                elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    white_minors += 1
                elif piece.piece_type == chess.ROOK:
                    white_majors += 1
                elif piece.piece_type == chess.QUEEN:
                    white_queens += 1
            else:
                black_material += value
                if piece.piece_type == chess.PAWN:
                    black_pawns += 1
                elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    black_minors += 1
                elif piece.piece_type == chess.ROOK:
                    black_majors += 1
                elif piece.piece_type == chess.QUEEN:
                    black_queens += 1
        
        # From player's perspective
        if is_white:
            material_balance = (white_material - black_material) / 100.0  # Normalize
            pawn_diff = white_pawns - black_pawns
            minor_diff = white_minors - black_minors
            major_diff = white_majors - black_majors
            queen_diff = white_queens - black_queens
        else:
            material_balance = (black_material - white_material) / 100.0
            pawn_diff = black_pawns - white_pawns
            minor_diff = black_minors - white_minors
            major_diff = black_majors - white_majors
            queen_diff = black_queens - white_queens
        
        return [material_balance, pawn_diff, minor_diff, major_diff, queen_diff]
    
    def _extract_king_safety(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract king safety features.
        
        Returns:
            [king_square_rank, pawn_shield_strength, king_activity]
        """
        king_square = board.king(is_white)
        if king_square is None:
            return [0.0, 0.0, 0.0]
        
        # King square rank (0-7, normalized to 0-1)
        # Higher rank for white = more advanced (riskier)
        # Lower rank for black = more advanced (riskier)
        rank = chess.square_rank(king_square)
        if is_white:
            king_rank = rank / 7.0  # 0 = back rank, 1 = advanced
        else:
            king_rank = (7 - rank) / 7.0
        
        # Pawn shield strength (count pawns protecting king)
        pawn_shield = 0
        king_file = chess.square_file(king_square)
        king_rank_pos = chess.square_rank(king_square)
        
        # Check adjacent files for pawns in front of king
        for file_offset in [-1, 0, 1]:
            file = king_file + file_offset
            if 0 <= file <= 7:
                if is_white:
                    # Look for pawns on ranks 2-3 in front of king
                    for rank_offset in [1, 2]:
                        rank = king_rank_pos + rank_offset
                        if 0 <= rank <= 7:
                            square = chess.square(file, rank)
                            if board.piece_at(square) == chess.Piece(chess.PAWN, chess.WHITE):
                                pawn_shield += 1
                else:
                    # Look for pawns on ranks 5-6 in front of king
                    for rank_offset in [1, 2]:
                        rank = king_rank_pos - rank_offset
                        if 0 <= rank <= 7:
                            square = chess.square(file, rank)
                            if board.piece_at(square) == chess.Piece(chess.PAWN, chess.BLACK):
                                pawn_shield += 1
        
        pawn_shield_strength = pawn_shield / 6.0  # Normalize (max ~6)
        
        # King activity (number of squares king can move to)
        king_activity = len(list(board.legal_moves)) if board.turn == is_white else 0
        # This is simplified - in practice, count squares around king
        
        return [king_rank, pawn_shield_strength, min(king_activity / 8.0, 1.0)]
    
    def _extract_pawn_structure(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract pawn structure features.
        
        Returns:
            [doubled_pawns, isolated_pawns, passed_pawns, pawn_advancement]
        """
        color = chess.WHITE if is_white else chess.BLACK
        player_pawns = board.pawns & board.occupied_co[color]
        enemy_pawns = board.pawns & board.occupied_co[not color]
        
        doubled = 0
        isolated = 0
        passed = 0
        total_rank = 0
        pawn_count = 0
        
        # Track pawns by file for doubled pawn detection
        pawns_by_file = defaultdict(list)
        
        for square in chess.scan_forward(player_pawns):
            if board.color_at(square) != color:
                continue
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            pawns_by_file[file].append(square)
            
            # Check for isolated pawns (no pawns on adjacent files)
            has_adjacent = False
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file <= 7:
                    if adj_file in pawns_by_file and len(pawns_by_file[adj_file]) > 0:
                        has_adjacent = True
                        break
            if not has_adjacent:
                isolated += 1
            
            # Check for passed pawns (no enemy pawns blocking)
            is_passed = True
            for enemy_square in chess.scan_forward(enemy_pawns):
                enemy_file = chess.square_file(enemy_square)
                enemy_rank = chess.square_rank(enemy_square)
                # Check if enemy pawn is in front or on adjacent files
                if abs(enemy_file - file) <= 1:
                    if is_white and enemy_rank > rank:
                        is_passed = False
                        break
                    elif not is_white and enemy_rank < rank:
                        is_passed = False
                        break
            if is_passed:
                passed += 1
            
            # Track advancement
            if is_white:
                total_rank += rank
            else:
                total_rank += (7 - rank)
            pawn_count += 1
        
        # Count doubled pawns (multiple pawns on same file)
        for file, squares in pawns_by_file.items():
            if len(squares) > 1:
                doubled += len(squares) - 1  # Count extra pawns
        
        # Normalize
        if pawn_count > 0:
            doubled_ratio = doubled / pawn_count
            isolated_ratio = isolated / pawn_count
            passed_ratio = passed / pawn_count
            avg_advancement = total_rank / (pawn_count * 7.0)
        else:
            doubled_ratio = 0.0
            isolated_ratio = 0.0
            passed_ratio = 0.0
            avg_advancement = 0.0
        
        return [doubled_ratio, isolated_ratio, passed_ratio, avg_advancement]
    
    def _extract_piece_activity(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract piece activity features.
        
        Returns:
            [total_moves, center_control, piece_coordination, development]
        """
        color = chess.WHITE if is_white else chess.BLACK
        pieces = board.occupied_co[color]
        
        total_moves = 0
        center_control = 0
        developed_pieces = 0
        total_pieces = 0
        
        # Center squares
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        for square in chess.scan_forward(pieces):
            piece = board.piece_at(square)
            if piece is None or piece.color != color:
                continue
            
            total_pieces += 1
            
            # Count legal moves for this piece
            # Simplified: count attacked squares
            attacks = board.attacks(square)
            move_count = len(attacks)
            total_moves += move_count
            
            # Center control
            center_bb = (chess.BB_SQUARES[center_squares[0]] | 
                        chess.BB_SQUARES[center_squares[1]] |
                        chess.BB_SQUARES[center_squares[2]] |
                        chess.BB_SQUARES[center_squares[3]])
            center_attacks = len(attacks & center_bb)
            center_control += center_attacks
            
            # Development (pieces off back rank)
            rank = chess.square_rank(square)
            if is_white and rank > 0:
                developed_pieces += 1
            elif not is_white and rank < 7:
                developed_pieces += 1
        
        # Normalize
        if total_pieces > 0:
            avg_moves = total_moves / (total_pieces * 20.0)  # Max ~20 moves per piece
            center_ratio = center_control / (total_pieces * 4.0)
            development_ratio = developed_pieces / total_pieces
        else:
            avg_moves = 0.0
            center_ratio = 0.0
            development_ratio = 0.0
        
        # Piece coordination (simplified: pieces supporting each other)
        coordination = min(avg_moves * 2, 1.0)  # Simplified metric
        
        return [avg_moves, center_ratio, coordination, development_ratio]
    
    def _get_game_phase_numeric(self, board: chess.Board) -> float:
        """
        Get game phase as numeric feature (0=opening, 1=endgame).
        
        Returns:
            Phase value between 0 and 1
        """
        piece_count = len(board.piece_map()) - 2  # Exclude kings
        # Opening: 30+ pieces, Endgame: <10 pieces
        phase = max(0, min(1, (30 - piece_count) / 20.0))
        return phase
    
    def _extract_control_features(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract board control features.
        
        Returns:
            [center_control, piece_mobility]
        """
        color = chess.WHITE if is_white else chess.BLACK
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        center_attacks = 0
        total_attacks = 0
        
        for square in chess.scan_forward(board.occupied_co[color]):
            attacks = board.attacks(square)
            total_attacks += len(attacks)
            
            for center_sq in center_squares:
                if center_sq in attacks:
                    center_attacks += 1
        
        # Normalize
        center_control = min(center_attacks / 20.0, 1.0)  # Max ~20 center attacks
        mobility = min(total_attacks / 100.0, 1.0)  # Max ~100 total attacks
        
        return [center_control, mobility]
    
    def _extract_tactical_features(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract tactical features (simplified).
        
        Returns:
            [hanging_pieces, tactical_opportunities]
        """
        color = chess.WHITE if is_white else chess.BLACK
        enemy_color = not color
        
        # Count hanging pieces (pieces attacked but not defended)
        hanging = 0
        for square in chess.scan_forward(board.occupied_co[color]):
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            attackers = board.attackers(enemy_color, square)
            defenders = board.attackers(color, square)
            
            if len(attackers) > len(defenders):
                hanging += 1
        
        # Tactical opportunities (pieces that can be captured)
        opportunities = 0
        for square in chess.scan_forward(board.occupied_co[enemy_color]):
            attackers = board.attackers(color, square)
            defenders = board.attackers(enemy_color, square)
            
            if len(attackers) > len(defenders):
                opportunities += 1
        
        # Normalize
        hanging_ratio = min(hanging / 16.0, 1.0)
        opportunities_ratio = min(opportunities / 16.0, 1.0)
        
        return [hanging_ratio, opportunities_ratio]
    
    def _extract_advanced_features(self, board: chess.Board, is_white: bool) -> List[float]:
        """
        Extract advanced sophisticated features.
        
        Returns:
            [piece_square_table_score, bishop_pair, rook_on_open_file, knight_outpost, piece_coordination_score]
        """
        color = chess.WHITE if is_white else chess.BLACK
        
        # Piece-square table scores (simplified)
        piece_square_score = 0
        bishop_count = 0
        rook_on_open_file = 0
        knight_outpost = 0
        coordination_score = 0
        
        # Basic piece-square values (center control bonus)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D4, 
                          chess.D5, chess.D6, chess.E3, chess.E4, chess.E5, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        for square in chess.scan_forward(board.occupied_co[color]):
            piece = board.piece_at(square)
            if piece is None or piece.color != color:
                continue
            
            # Piece-square table (simplified)
            if square in center_squares:
                piece_square_score += 0.1
            elif square in extended_center:
                piece_square_score += 0.05
            
            # Bishop pair
            if piece.piece_type == chess.BISHOP:
                bishop_count += 1
            
            # Rook on open file
            if piece.piece_type == chess.ROOK:
                file = chess.square_file(square)
                file_bb = chess.BB_FILES[file]
                # Check if file has no pawns (bitboard is integer, check if zero)
                if (board.pawns & file_bb) == 0:
                    rook_on_open_file += 1
            
            # Knight outpost (simplified: knight on advanced square with pawn support)
            if piece.piece_type == chess.KNIGHT:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                if is_white and rank >= 4:
                    # Check for pawn support
                    support_squares = [chess.square(file-1, rank-1), chess.square(file+1, rank-1)]
                    if any(board.piece_at(sq) == chess.Piece(chess.PAWN, color) for sq in support_squares if 0 <= sq < 64):
                        knight_outpost += 1
                elif not is_white and rank <= 3:
                    support_squares = [chess.square(file-1, rank+1), chess.square(file+1, rank+1)]
                    if any(board.piece_at(sq) == chess.Piece(chess.PAWN, color) for sq in support_squares if 0 <= sq < 64):
                        knight_outpost += 1
        
        # Bishop pair bonus
        bishop_pair = 1.0 if bishop_count >= 2 else 0.0
        
        # Normalize
        piece_square_score = min(piece_square_score, 1.0)
        rook_on_open_file = min(rook_on_open_file / 2.0, 1.0)  # Max 2 rooks
        knight_outpost = min(knight_outpost / 2.0, 1.0)  # Max 2 knights
        
        # Piece coordination (pieces attacking same squares)
        coordination_count = 0
        for square in chess.scan_forward(board.occupied_co[not color]):
            attackers = list(board.attackers(color, square))
            if len(attackers) > 1:
                coordination_count += len(attackers) - 1
        coordination_score = min(coordination_count / 10.0, 1.0)
        
        return [piece_square_score, bishop_pair, rook_on_open_file, knight_outpost, coordination_score]
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability."""
        return [
            # Material (5)
            'material_balance', 'pawn_diff', 'minor_piece_diff', 'major_piece_diff', 'queen_diff',
            # King safety (3)
            'king_rank', 'pawn_shield', 'king_activity',
            # Pawn structure (4)
            'doubled_pawns', 'isolated_pawns', 'passed_pawns', 'pawn_advancement',
            # Piece activity (4)
            'piece_mobility', 'center_control', 'piece_coordination', 'development',
            # Game phase (1)
            'game_phase',
            # Control (2)
            'center_control', 'board_mobility',
            # Tactical (2)
            'hanging_pieces', 'tactical_opportunities',
            # Advanced (5)
            'piece_square_score', 'bishop_pair', 'rook_open_file', 'knight_outpost', 'coordination_score'
        ]

