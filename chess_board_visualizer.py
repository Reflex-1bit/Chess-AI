"""
Alternative Chess Board Visualizer

Provides fallback visualization methods when chess.svg is not available.
"""

import chess
import streamlit as st
from typing import Optional, List, Tuple
from io import StringIO


def render_board_unicode(board: chess.Board) -> str:
    """
    Render chess board using Unicode characters as a fallback.
    
    Args:
        board: Chess board position
        
    Returns:
        Unicode string representation of the board
    """
    unicode_pieces = {
        'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
    }
    
    result = "\n"
    result += "  " + " ".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) + "\n"
    result += "  " + "─" * 31 + "\n"
    
    for rank in range(7, -1, -1):
        line = f"{rank+1}│"
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                unicode_char = unicode_pieces.get(symbol, symbol)
                line += f" {unicode_char} "
            else:
                # Alternate square colors
                if (rank + file) % 2 == 0:
                    line += " · "
                else:
                    line += "   "
        result += line + f"│{rank+1}\n"
    
    result += "  " + "─" * 31 + "\n"
    result += "  " + " ".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) + "\n"
    
    return result


def render_board_html_fallback(board: chess.Board, size: int = 640, interactive: bool = False) -> str:
    """
    Render chess board using HTML/CSS as a visual fallback.
    
    Args:
        board: Chess board position
        size: Board size in pixels (default 600 for better visibility)
        interactive: If True, enables drag-and-drop functionality
        
    Returns:
        HTML string with styled board
    """
    square_size = size // 8
    
    # Use larger, clearer Unicode piece symbols
    piece_symbols = {
        'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',  # Black pieces (outline)
        'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕', 'k': '♔', 'p': '♙'   # White pieces (filled)
    }
    
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    # Better color scheme - higher contrast
    light_square = '#f0d9b5'  # Light beige
    dark_square = '#b58863'   # Brown
    
    html = f'''
    <div id="chess-board-container" style="display: flex; justify-content: center; margin: 20px 0; padding: 20px; background-color: #f8f9fa; width: 100%; box-sizing: border-box; overflow-x: auto;">
        <div id="chess-board" style="width: {size}px; height: {size}px; min-width: {size}px; border: 4px solid #2c3e50; display: grid; grid-template-columns: repeat(8, 1fr); background-color: #2c3e50; box-shadow: 0 4px 8px rgba(0,0,0,0.3); border-radius: 4px; flex-shrink: 0;">
    '''
    
    for rank in range(7, -1, -1):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            square_name = chess.square_name(square)
            
            # Square color (light/dark)
            is_light = (rank + file) % 2 == 0
            bg_color = light_square if is_light else dark_square
            
            # Piece symbol and color - much better contrast
            piece_char = ''
            piece_color = '#000000'  # Black for maximum visibility
            piece_type = ''
            piece_color_name = ''
            
            if piece:
                symbol = piece.symbol()
                piece_char = piece_symbols.get(symbol, symbol)
                piece_type = piece.symbol().upper()
                piece_color_name = 'white' if piece.color == chess.WHITE else 'black'
                # Use very dark colors for both white and black pieces for better visibility
                if piece.color == chess.WHITE:
                    piece_color = '#000000'  # Pure black for white pieces (filled symbols show better)
                else:
                    piece_color = '#1a1a1a'  # Very dark gray for black pieces (outline symbols)
            
            # Add coordinates on edge squares - larger and more visible
            coord_label = ''
            coord_font_size = max(12, int(square_size * 0.2))  # Larger coordinate font
            if rank == 7:  # Top rank - show file letters
                coord_label = f'<div style="position: absolute; bottom: 3px; right: 3px; font-size: {coord_font_size}px; font-weight: bold; color: {"#000" if is_light else "#fff"}; text-shadow: 1px 1px 2px rgba(255,255,255,0.8);">{files[file]}</div>'
            if file == 0:  # Leftmost file - show rank numbers
                coord_label += f'<div style="position: absolute; top: 3px; left: 3px; font-size: {coord_font_size}px; font-weight: bold; color: {"#000" if is_light else "#fff"}; text-shadow: 1px 1px 2px rgba(255,255,255,0.8);">{ranks[rank]}</div>'
            
            # Larger font size for pieces - use 80% of square size for maximum visibility
            piece_font_size = int(square_size * 0.80)
            
            # Create piece HTML if piece exists
            piece_html = ''
            if piece_char:
                draggable_attr = 'draggable="true"' if interactive else ''
                piece_html = f'''
                <div class="chess-piece" 
                     data-square="{square_name}"
                     data-piece="{piece_type}"
                     data-color="{piece_color_name}"
                     {draggable_attr}
                     style="font-size: {piece_font_size}px; 
                            font-weight: 900; 
                            color: {piece_color};
                            cursor: {'grab' if interactive else 'default'};
                            user-select: none;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.5), -1px -1px 2px rgba(255,255,255,0.8);
                            transition: transform 0.1s;
                            z-index: 10;">
                    {piece_char}
                </div>
            '''
            
            html += f'''
            <div class="square" 
                 data-square="{square_name}"
                 data-file="{file}"
                 data-rank="{rank}"
                 style="background-color: {bg_color}; 
                        width: {square_size}px; 
                        height: {square_size}px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        position: relative; 
                        border: 1px solid rgba(0,0,0,0.3);
                        cursor: {'pointer' if interactive else 'default'};">
                {piece_html}
                {coord_label}
            </div>
            '''
    
    html += '''
        </div>
        <div id="move-result" style="margin-top: 10px; text-align: center; font-weight: bold; color: #2c3e50; display: none;"></div>
    </div>
    '''
    
    # Add JavaScript for drag and drop if interactive
    if interactive:
        js_code = '''
    <script>
    (function() {
        let draggedPiece = null;
        let startSquare = null;
        const board = document.getElementById('chess-board');
        if (!board) return;
        
        const pieces = board.querySelectorAll('.chess-piece');
        const squares = board.querySelectorAll('.square');
        
        pieces.forEach(piece => {
            piece.addEventListener('dragstart', function(e) {
                draggedPiece = this;
                startSquare = this.parentElement.dataset.square;
                this.style.opacity = '0.5';
                e.dataTransfer.effectAllowed = 'move';
                
                squares.forEach(sq => {
                    if (sq !== this.parentElement) {
                        const isLight = parseInt(sq.dataset.file) % 2 === parseInt(sq.dataset.rank) % 2;
                        sq.style.backgroundColor = isLight ? '#d4c4a0' : '#c19a6b';
                        sq.style.border = '2px solid #3b82f6';
                    }
                });
            });
            
            piece.addEventListener('dragend', function(e) {
                this.style.opacity = '1';
                squares.forEach(sq => {
                    const isLight = parseInt(sq.dataset.file) % 2 === parseInt(sq.dataset.rank) % 2;
                    sq.style.backgroundColor = isLight ? '#f0d9b5' : '#b58863';
                    sq.style.border = '1px solid rgba(0,0,0,0.3)';
                });
                draggedPiece = null;
                startSquare = null;
            });
        });
        
        squares.forEach(square => {
            square.addEventListener('dragover', function(e) {
                if (draggedPiece) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    const isLight = parseInt(this.dataset.file) % 2 === parseInt(this.dataset.rank) % 2;
                    this.style.backgroundColor = isLight ? '#a8d8ea' : '#8bc34a';
                }
            });
            
            square.addEventListener('dragleave', function(e) {
                if (draggedPiece && this !== draggedPiece.parentElement) {
                    const isLight = parseInt(this.dataset.file) % 2 === parseInt(this.dataset.rank) % 2;
                    this.style.backgroundColor = isLight ? '#d4c4a0' : '#c19a6b';
                }
            });
            
            square.addEventListener('drop', function(e) {
                e.preventDefault();
                if (draggedPiece && startSquare) {
                    const endSquare = this.dataset.square;
                    const move = startSquare + endSquare;
                    
                    // Update board visually
                    if (this.querySelector('.chess-piece')) {
                        this.removeChild(this.querySelector('.chess-piece'));
                    }
                    const newPiece = draggedPiece.cloneNode(true);
                    newPiece.style.opacity = '1';
                    this.appendChild(newPiece);
                    if (draggedPiece.parentElement) {
                        draggedPiece.parentElement.removeChild(draggedPiece);
                    }
                    
                    // Store move for Streamlit - try multiple methods
                    if (window.parent) {
                        window.parent.postMessage({
                            type: 'chess-move',
                            move: move
                        }, '*');
                    }
                    
                    // Also try localStorage
                    try {
                        localStorage.setItem('chessMove', move);
                        localStorage.setItem('chessMoveTime', Date.now().toString());
                    } catch(e) {}
                    
                    // Show move result
                    const moveResult = document.getElementById('move-result');
                    if (moveResult) {
                        moveResult.textContent = 'Move: ' + move;
                        moveResult.style.display = 'block';
                        moveResult.style.color = '#22c55e';
                    }
                    
                    // Reset highlights
                    squares.forEach(sq => {
                        const isLight = parseInt(sq.dataset.file) % 2 === parseInt(sq.dataset.rank) % 2;
                        sq.style.backgroundColor = isLight ? '#f0d9b5' : '#b58863';
                        sq.style.border = '1px solid rgba(0,0,0,0.3)';
                    });
                    
                    draggedPiece = null;
                    startSquare = null;
                }
            });
        });
    })();
    </script>
    '''
        html += js_code
    
    return html
