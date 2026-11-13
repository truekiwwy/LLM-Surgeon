<#
.SYNOPSIS
  Builds the Windows distribution: portable ZIP, per-user NSIS installer, SHA256SUMS.

.DESCRIPTION
  Run from anywhere; paths are resolved from this script. Output goes to dist\.
  -DryRun builds nothing with cargo: it stages placeholder executables and compiles the
  installer, which validates packaging\windows\installer.nsi (used by CI on every change).
  Code signing is optional: with SIGN_CERT_BASE64 (a base64 .pfx) and SIGN_CERT_PASSWORD in
  the environment the executables and the installer are signed; without them, they are not.
#>
param(
  [string]$Version,
  [string]$Out = "dist",
  [switch]$DryRun,
  [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$gui = Join-Path $root "crates\surgeon-gui"
$out = Join-Path $root $Out
$stage = Join-Path $out "stage"

if (-not $Version) {
  $Version = (Select-String -Path (Join-Path $gui "Cargo.toml") -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
}
Write-Host "version $Version"
Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stage | Out-Null

function Invoke-Native([scriptblock]$Command) {
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "command failed ($LASTEXITCODE): $Command" }
}

# --- binaries ---------------------------------------------------------------------------
if ($DryRun) {
  "placeholder" | Set-Content (Join-Path $stage "surgeon-gui.exe")
  "placeholder" | Set-Content (Join-Path $stage "surgeon.exe")
} else {
  if (-not $SkipBuild) {
    # Static CRT comes from crates\surgeon-gui\.cargo\config.toml: build from that folder.
    Push-Location $root;  try { Invoke-Native { cargo build --release --locked --bin surgeon } } finally { Pop-Location }
    Push-Location $gui;   try { Invoke-Native { cargo build --release --locked } } finally { Pop-Location }
  }
  Copy-Item (Join-Path $gui "target\release\surgeon-gui.exe") $stage
  Copy-Item (Join-Path $root "target\release\surgeon.exe") $stage
}
Copy-Item (Join-Path $PSScriptRoot "windows\README-WINDOWS.txt") $stage

# --- optional signing -------------------------------------------------------------------
$signed = $false
$pfx = $null
function Invoke-Sign([string[]]$Files) {
  if (-not $script:pfx) { return }
  $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
    Sort-Object FullName | Select-Object -Last 1
  if (-not $signtool) { throw "signtool.exe not found (Windows SDK)" }
  foreach ($f in $Files) {
    Invoke-Native { & $signtool.FullName sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /f $script:pfx /p $env:SIGN_CERT_PASSWORD $f }
  }
}
if (-not $DryRun -and $env:SIGN_CERT_BASE64 -and $env:SIGN_CERT_PASSWORD) {
  $pfx = Join-Path $env:TEMP "surgeon-sign.pfx"
  [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:SIGN_CERT_BASE64))
  $signed = $true
  Invoke-Sign @((Join-Path $stage "surgeon-gui.exe"), (Join-Path $stage "surgeon.exe"))
} else {
  Write-Host "no signing certificate in the environment: binaries are NOT signed"
}

# --- portable zip -----------------------------------------------------------------------
$base = "weight-surgeon-$Version-windows-x64"
$zipDir = Join-Path $out $base
New-Item -ItemType Directory -Force $zipDir | Out-Null
Copy-Item (Join-Path $stage "*") $zipDir
"Portable mode: this file makes the app keep everything in .\data" | Set-Content (Join-Path $zipDir "portable.txt")
Compress-Archive -Path $zipDir -DestinationPath (Join-Path $out "$base.zip")
Remove-Item $zipDir -Recurse -Force

# --- installer --------------------------------------------------------------------------
$makensis = (Get-Command makensis -ErrorAction SilentlyContinue).Source
if (-not $makensis) { $makensis = "${env:ProgramFiles(x86)}\NSIS\makensis.exe" }
if (-not (Test-Path $makensis)) { throw "makensis not found: install NSIS (choco install nsis)" }
$setup = Join-Path $out "$base-setup.exe"
Invoke-Native { & $makensis /V2 "/DVERSION=$Version" "/DSTAGE=$stage" "/DOUTFILE=$setup" (Join-Path $PSScriptRoot "windows\installer.nsi") }
Invoke-Sign @($setup)
if ($pfx) { Remove-Item $pfx -Force }

# --- checksums --------------------------------------------------------------------------
Remove-Item $stage -Recurse -Force
Get-ChildItem $out -File | Sort-Object Name | ForEach-Object {
  "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name
} | Set-Content (Join-Path $out "SHA256SUMS") -Encoding ascii
Get-Content (Join-Path $out "SHA256SUMS")
Write-Host ("signed: {0}" -f $signed)
