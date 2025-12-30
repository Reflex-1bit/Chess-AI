# How to Open/View Your GitHub Repository

## Option 1: View Repository on GitHub Website (After Pushing)

Once you've pushed your code to GitHub, you can view it by:

1. **Go to GitHub.com** in your web browser
2. **Sign in** to your account
3. **Click your profile picture** (top right) → **Your repositories**
4. **Click on your repository name** (e.g., `chess-coaching-system`)
5. You'll see all your files, README, and can browse the code

**Direct URL format:**
```
https://github.com/YOUR_USERNAME/chess-coaching-system
```
(Replace `YOUR_USERNAME` with your actual GitHub username)

---

## Option 2: Open Local Repository in VS Code

To work on the code locally:

### Method A: From VS Code
1. Open **VS Code**
2. Go to **File** → **Open Folder**
3. Navigate to: `C:\chess_coaching_system`
4. Click **Select Folder**

### Method B: From Command Line
1. Open **Command Prompt** or **PowerShell**
2. Navigate to the project:
   ```bash
   cd C:\chess_coaching_system
   ```
3. Open in VS Code:
   ```bash
   code .
   ```

### Method C: Right-Click Menu
1. Navigate to `C:\chess_coaching_system` in Windows Explorer
2. **Right-click** in the folder
3. Select **"Open with Code"** (if you installed the VS Code context menu option)

---

## Option 3: Clone Repository (If Working on Multiple Computers)

If you want to download your GitHub repository to another computer:

1. **Copy the repository URL** from GitHub:
   - Go to your repository on GitHub
   - Click the green **"Code"** button
   - Copy the HTTPS URL (e.g., `https://github.com/YOUR_USERNAME/chess-coaching-system.git`)

2. **Open terminal/command prompt** on the other computer

3. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chess-coaching-system.git
   cd chess-coaching-system
   ```

---

## Option 4: Quick Links (After Repository is Created)

Once your repository is on GitHub, you can:

- **View code**: `https://github.com/YOUR_USERNAME/chess-coaching-system`
- **View commits**: `https://github.com/YOUR_USERNAME/chess-coaching-system/commits/main`
- **View issues**: `https://github.com/YOUR_USERNAME/chess-coaching-system/issues`
- **Settings**: `https://github.com/YOUR_USERNAME/chess-coaching-system/settings`

---

## Running the Application Locally

To run the Streamlit app from your local repository:

1. **Open terminal** in the project directory:
   ```bash
   cd C:\chess_coaching_system
   ```

2. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

4. The app will open in your browser at `http://localhost:8501`

---

## Need Help?

- **GitHub Help**: [docs.github.com](https://docs.github.com)
- **Git Tutorial**: [git-scm.com/docs](https://git-scm.com/docs)
- **VS Code Git Integration**: [code.visualstudio.com/docs/sourcecontrol/overview](https://code.visualstudio.com/docs/sourcecontrol/overview)

