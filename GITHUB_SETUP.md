# Setting Up GitHub Repository

This guide will help you create a GitHub repository for the Chess Coaching System.

## Prerequisites

1. **Install Git** (if not already installed):
   - Windows: Download from [git-scm.com](https://git-scm.com/download/win)
   - Mac: `brew install git` or download from git-scm.com
   - Linux: `sudo apt install git` (Ubuntu/Debian) or `sudo yum install git` (RHEL/CentOS)

2. **Create a GitHub account** (if you don't have one):
   - Go to [github.com](https://github.com) and sign up

## Steps to Create and Push to GitHub

### 1. Initialize Git Repository (if not already done)

Open a terminal/command prompt in the project directory and run:

```bash
git init
```

### 2. Configure Git (if first time)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: Chess Coaching System with ML-based mistake analysis"
```

### 5. Create Repository on GitHub

1. Go to [github.com](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `chess-coaching-system` (or your preferred name)
5. Description: "ML-based chess coaching system that analyzes games and provides personalized insights"
6. Choose **Public** or **Private**
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

### 6. Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/chess-coaching-system.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/chess-coaching-system.git
```

### 7. Push to GitHub

```bash
# Push to GitHub (first time)
git push -u origin main

# If you get an error about 'main' branch, try 'master':
# git push -u origin master
```

### 8. Verify and Open Repository

Go to your GitHub repository page and verify all files are there!

**To view your repository:**
- Direct URL: `https://github.com/YOUR_USERNAME/chess-coaching-system`
- Or: Go to GitHub.com → Your Profile → Repositories → Click your repo name

**To open locally in VS Code:**
```bash
cd C:\chess_coaching_system
code .
```

## Future Updates

After making changes to files:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## What Gets Ignored

The `.gitignore` file ensures these files/folders are NOT uploaded:
- `__pycache__/` - Python cache files
- `uploads/` - User uploaded files
- `models/` - Trained model files (optional)
- `build/`, `dist/` - Build artifacts
- `*.pkl` - Pickle files (model files)
- `.streamlit/` - Streamlit config
- Virtual environments

## Adding a License

Consider adding a license file:
- MIT License (permissive, common for open source)
- Apache 2.0 License
- GPL License (copyleft)

You can add one from GitHub when creating the repo, or create a `LICENSE` file manually.

## Repository Description Template

When creating the GitHub repository, you can use this description:

```
ML-based chess coaching system that analyzes player games, identifies recurring weaknesses, and provides personalized coaching insights. Features interactive board analysis, blunder detection, puzzle recommendations, and comprehensive visualizations.
```

## Tags/Topics to Add on GitHub

After creating the repository, add these topics:
- `chess`
- `machine-learning`
- `python`
- `streamlit`
- `stockfish`
- `chess-analysis`
- `coaching`
- `data-science`

