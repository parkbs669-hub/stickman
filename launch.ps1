$ErrorActionPreference = 'Stop'
$launcherPath = Join-Path $PSScriptRoot 'stickman_launcher.py'
$candidates = @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
    'C:\Python313\python.exe'
)
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) { $candidates += $pythonCommand.Source }
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $candidate) {
        try {
            & $candidate -c 'import cv2, numpy, mediapipe, PIL, tkinter' 2>$null
            if ($LASTEXITCODE -eq 0) {
                & $candidate $launcherPath
                exit $LASTEXITCODE
            }
        } catch { continue }
    }
}
Write-Host 'No Python with required packages found. See README.md (Python 3.13).'
Read-Host 'Press Enter to close'
exit 1
