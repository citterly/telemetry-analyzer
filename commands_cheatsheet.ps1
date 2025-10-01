<#
================================================================================
 Windows Command Cheat Sheet — Telemetry Analyzer Project
--------------------------------------------------------------------------------
 This file is NOT meant to be executed as a script. 
 Keep it in your project as a quick reference for common commands.
================================================================================
#>

# ------------------------------------------------------------------------------
# 📂 Directory & File Listings
# ------------------------------------------------------------------------------
dir /b                   # List files only (bare format)
dir /s                   # List with subdirectories
tree /f /a               # Show tree with files (ASCII only, safe for sharing)
tree /f /a > tree.txt    # Save full tree to a text file

# ------------------------------------------------------------------------------
# 🔎 Search in Files
# ------------------------------------------------------------------------------
findstr /s /i "XRK" *.py        # Search recursively for "XRK" in all .py files
findstr /n "def " file.py       # List functions with line numbers in file.py

# ------------------------------------------------------------------------------
# 🧹 Cleanup
# ------------------------------------------------------------------------------
del /s /q *.pyc                 # Delete all Python bytecode files
del /s /q *.log                 # Delete all log files
rmdir /s /q build               # Remove "build" directory recursively

# ------------------------------------------------------------------------------
# 📦 Hashes & File Comparison
# ------------------------------------------------------------------------------
certutil -hashfile myfile.py SHA256   # Get SHA256 hash of a file
certutil -hashfile myfile.py MD5      # Get MD5 hash
fc file1.py file2.py                  # Compare two files (text mode)
fc /b file1 file2                     # Compare two files (binary mode)

# ------------------------------------------------------------------------------
# ⚙️ System Info & Processes
# ------------------------------------------------------------------------------
tasklist                               # List running processes
taskkill /IM python.exe /F             # Kill all Python processes
systeminfo                             # Show system summary
echo %PATH%                            # Show PATH variable

# ------------------------------------------------------------------------------
# 🖥️ Python & Environment
# ------------------------------------------------------------------------------
where python                           # Show which Python executable is used
python -m venv venv                    # Create virtual environment
pip freeze > requirements.txt          # Export installed packages

# ------------------------------------------------------------------------------
# 🔧 Project Utilities
# ------------------------------------------------------------------------------
# Run project tree generator (defined separately in project_tree.ps1)
.\project_tree.ps1

# Example: run FileManager CLI commands
python file_manager.py import data\uploads\mysession.xrk
python file_manager.py list
python file_manager.py process mysession.xrk
python file_manager.py stats
