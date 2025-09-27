<#
.SYNOPSIS
    Cleanup AIM SDK units.xml redirection files

.DESCRIPTION
    Removes fallback copies of units.xml from:
    - SDK hard-coded path (C:\Python313\user\profiles)
    - Project-local aim_cache

    Use this if you want to reset and re-run ensure_units_redirect()
#>

Write-Host "🧹 Cleaning AIM SDK redirection files..." -ForegroundColor Cyan

# Paths
$projectCache = Join-Path $PSScriptRoot "..\data\exports\aim_cache\units.xml"
$sdkProfile   = "C:\Python313\user\profiles\units.xml"

# Remove project cache copy
if (Test-Path $projectCache) {
    Remove-Item $projectCache -Force
    Write-Host "✅ Removed project-local units.xml: $projectCache"
} else {
    Write-Host "ℹ️ No project-local units.xml found"
}

# Remove SDK profile copy
if (Test-Path $sdkProfile) {
    Remove-Item $sdkProfile -Force
    Write-Host "✅ Removed SDK path units.xml: $sdkProfile"
} else {
    Write-Host "ℹ️ No SDK path units.xml found"
}

Write-Host "🧹 Cleanup complete" -ForegroundColor Green
