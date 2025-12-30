"""
Chess.com API Integration

Fetches games directly from Chess.com using their public API.
"""

import requests
import chess.pgn
from io import StringIO
from typing import List, Dict, Optional
import time


class ChessComAPI:
    """Interface to Chess.com API for fetching games."""
    
    BASE_URL = "https://api.chess.com/pub"
    
    def __init__(self):
        """Initialize Chess.com API client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Chess-Coaching-System/1.0'
        })
    
    def get_player_games(self, username: str, year: Optional[int] = None, month: Optional[int] = None) -> List[str]:
        """
        Get PGN strings for a player's games.
        
        Args:
            username: Chess.com username
            year: Year (YYYY), if None gets current year
            month: Month (1-12), if None gets current month
        
        Returns:
            List of PGN strings
        """
        if year is None:
            from datetime import datetime
            year = datetime.now().year
        if month is None:
            from datetime import datetime
            month = datetime.now().month
        
        # Format month as two digits
        month_str = f"{month:02d}"
        
        url = f"{self.BASE_URL}/player/{username}/games/{year}/{month_str}/pgn"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # PGN data is returned as text, games are separated by empty lines
            pgn_text = response.text
            
            if not pgn_text or not pgn_text.strip():
                return []
            
            # Chess.com returns multiple games separated by blank lines
            # Each game starts with [Event and ends before the next [Event or end of text
            games = []
            
            # Split by [Event tags - this is more reliable than blank lines
            parts = pgn_text.split('[Event')
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Re-add the [Event prefix (except for first part if it doesn't start with it)
                if i == 0 and not part.startswith('['):
                    # First part might not be a game, skip if it doesn't look like PGN
                    if not any(tag in part for tag in ['[White', '[Black', '[Result']):
                        continue
                    # It might be a partial game, try to prepend [Event
                    game_text = '[Event' + part
                else:
                    game_text = '[Event' + part if i > 0 else part
                
                game_text = game_text.strip()
                
                # Only add if it looks like a valid game (has headers and potentially moves)
                if game_text and len(game_text) > 50:
                    # Check if it has at least some PGN structure
                    if '[White' in game_text or '[Black' in game_text:
                        games.append(game_text)
            
            return games
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch games from Chess.com: {e}")
    
    def get_player_archives(self, username: str) -> List[str]:
        """
        Get list of available archive URLs for a player.
        
        Args:
            username: Chess.com username
        
        Returns:
            List of archive URLs
        """
        url = f"{self.BASE_URL}/player/{username}/games/archives"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('archives', [])
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch archives from Chess.com: {e}")
    
    def get_games_from_archive(self, archive_url: str) -> List[str]:
        """
        Get PGN strings from an archive URL.
        
        Args:
            archive_url: URL to archive (e.g., https://api.chess.com/pub/player/{username}/games/2024/01/pgn)
        
        Returns:
            List of PGN strings
        """
        try:
            response = self.session.get(archive_url, timeout=10)
            response.raise_for_status()
            
            pgn_text = response.text
            
            if not pgn_text or not pgn_text.strip():
                return []
            
            # Split into individual games (same logic as get_player_games)
            games = []
            
            # Split by [Event tags
            parts = pgn_text.split('[Event')
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Re-add the [Event prefix
                if i == 0 and not part.startswith('['):
                    if not any(tag in part for tag in ['[White', '[Black', '[Result']):
                        continue
                    game_text = '[Event' + part
                else:
                    game_text = '[Event' + part if i > 0 else part
                
                game_text = game_text.strip()
                
                if game_text and len(game_text) > 50:
                    if '[White' in game_text or '[Black' in game_text:
                        games.append(game_text)
            
            return games
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch games from archive: {e}")
    
    def get_all_recent_games(self, username: str, months: int = 3) -> List[str]:
        """
        Get all games from the last N months.
        
        Args:
            username: Chess.com username
            months: Number of months to fetch (default: 3)
        
        Returns:
            List of PGN strings
        """
        from datetime import datetime, timedelta
        
        all_games = []
        current_date = datetime.now()
        
        for i in range(months):
            target_date = current_date - timedelta(days=30 * i)
            year = target_date.year
            month = target_date.month
            
            try:
                games = self.get_player_games(username, year, month)
                all_games.extend(games)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Warning: Could not fetch games for {year}/{month}: {e}")
                continue
        
        return all_games
    
    def validate_username(self, username: str) -> bool:
        """
        Check if a Chess.com username exists.
        
        Args:
            username: Chess.com username
        
        Returns:
            True if username exists, False otherwise
        """
        url = f"{self.BASE_URL}/player/{username}"
        
        try:
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

