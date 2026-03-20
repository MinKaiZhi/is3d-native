$ErrorActionPreference = 'Stop'

function Initialize-PytestWorkspace {
    param(
        [string[]]$Candidates,
        [string]$RunId
    )

    foreach ($candidate in $Candidates) {
        try {
            $root = [System.IO.Path]::GetFullPath($candidate)
            $tempRoot = Join-Path $root ("temp_" + $RunId)
            $baseTemp = Join-Path $root ("basetemp_" + $RunId)

            New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
            New-Item -ItemType Directory -Force -Path $baseTemp | Out-Null

            return @{
                Root = $root
                TempRoot = $tempRoot
                BaseTemp = $baseTemp
            }
        }
        catch {
            Write-Host "[WARN] Cannot use pytest temp root: $candidate" -ForegroundColor Yellow
            Write-Host "       $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }

    throw 'No writable temp directory available for pytest.'
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "python not found in PATH. Activate your conda env before running scripts/test.ps1"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$localAppData = [Environment]::GetFolderPath('LocalApplicationData')
$runId = [DateTime]::Now.ToString('yyyyMMdd_HHmmss_fff')

$paths = @(
    (Join-Path $localAppData 'Temp\is3d-native-pytest'),
    (Join-Path $repoRoot '.pytest_tmp_local')
)

$workspace = Initialize-PytestWorkspace -Candidates $paths -RunId $runId

$env:TEMP = $workspace.TempRoot
$env:TMP = $workspace.TempRoot

& $python.Source -m pytest -q --basetemp $workspace.BaseTemp -p no:cacheprovider
