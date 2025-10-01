# Generate a clean project tree for sharing
# Usage: Run with PowerShell from the project root

$OutputFile = "project_tree.txt"
$IgnorePatterns = @(".git", "venv", "__pycache__", ".pytest_cache", "node_modules", ".vscode", ".idea")

Write-Host "Generating clean project tree..."
Write-Host ("Ignoring: " + ($IgnorePatterns -join ", "))

# Generate ASCII tree with files
$Tree = tree /f /a

# Filter out ignored patterns
foreach ($pattern in $IgnorePatterns) {
    $Tree = $Tree | Select-String -Pattern $pattern -NotMatch
}

# Save to file
$Tree | Out-File -Encoding ascii $OutputFile

Write-Host ("Project tree saved to " + $OutputFile)
