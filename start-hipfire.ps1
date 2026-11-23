# Starts the hipfire daemon in WSL2 in the background and waits until it answers.
# Usage: powershell -ExecutionPolicy Bypass -File .\start-hipfire.ps1 [-User <wsl user>]
# -User is the WSL account whose ~/.hipfire holds the models; it defaults to the Windows one,
# which is usually the account WSL created at install time.
param(
    [string]$User = $env:USERNAME
)
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$distro = 'Ubuntu-24.04'
$health = 'http://127.0.0.1:11435/health'
$script = "/home/$User/start-hipfire.sh"
$log = "/home/$User/hipfire-serve.log"
$timeoutSeconds = 60

function Test-Daemon {
    try {
        (Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200
    } catch {
        $false
    }
}

if (Test-Daemon) {
    Write-Output "hipfire already answers on $health"
    exit 0
}

# The daemon script is not in the distribution until copied there from this repository.
wsl.exe -d $distro -u $User -e test -x $script
if ($LASTEXITCODE -ne 0) {
    $source = Join-Path $PSScriptRoot 'start-hipfire.sh'
    $linux = wsl.exe -d $distro -u $User -e wslpath -a $source
    if ($LASTEXITCODE -ne 0) { $linux = $source }
    [Console]::Error.WriteLine("$script is missing or not executable in $distro for user '$User' (or that user or distribution does not exist).")
    [Console]::Error.WriteLine("Install it from this repository, then run this script again:")
    [Console]::Error.WriteLine("  wsl.exe -d $distro -u $User -e install -m 755 $linux $script")
    [Console]::Error.WriteLine("Another WSL user: -User <name>.")
    exit 2
}

# This hidden wsl.exe stays attached: it keeps the distro up as long as the daemon runs.
$arguments = "-d $distro -u $User -- $script > $log 2>&1"
$wsl = Start-Process wsl.exe -ArgumentList $arguments -WindowStyle Hidden -PassThru

$deadline = (Get-Date).AddSeconds($timeoutSeconds)
while ((Get-Date) -lt $deadline -and -not $wsl.HasExited) {
    if (Test-Daemon) {
        Write-Output "hipfire is ready on $health (log: $log in $distro)"
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

Write-Output "hipfire did not answer on $health. Last lines of ${log}:"
wsl.exe -d $distro -u $User -- tail -n 20 $log
exit 1
