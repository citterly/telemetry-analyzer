param(
    [string]$Target = "C:\Users\chris\source\repos\telemetry-analyzer\data\exports\aim_cache",
    [string]$Link   = "C:\Python313\user\profiles"
)

Write-Host "Setting up AIM DLL redirect..."
Write-Host "Link:   $Link"
Write-Host "Target: $Target"

# Remove existing link if it exists
if (Test-Path $Link) {
    Write-Host "Removing existing path at $Link"
    Remove-Item $Link -Recurse -Force
}

# Create new junction
Write-Host "Creating junction..."
New-Item -ItemType Junction -Path $Link -Target $Target

Write-Host "✅ Redirect complete! AIM SDK will now write to $Target"
