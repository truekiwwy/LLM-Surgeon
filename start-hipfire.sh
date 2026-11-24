#!/bin/bash
# Runs the hipfire daemon for Weight Surgeon in the foreground, as the WSL user whose
# ~/.hipfire holds the models (start-hipfire.ps1 starts it in the background).
# Install it in the distribution (the error of start-hipfire.ps1 prints this line for you):
#   wsl.exe -d Ubuntu-24.04 -u <user> -e install -m 755 /mnt/c/<repo>/start-hipfire.sh /home/<user>/start-hipfire.sh
# Set for the RX 7600 XT (gfx1102), outside ROCm's WSL matrix, which also needs the line
# 0x7480,11,0,2 in /opt/rocm/share/rocdxg/dids.conf. On a card of the matrix, drop LD_PRELOAD
# and --spec.
export PATH="$HOME/.hipfire/bin:/opt/rocm/bin:$PATH"
# ROCm releases before 7.13 only look for /dev/dxg when asked to.
export HSA_ENABLE_DXG_DETECTION=1
# Optional: the gfx1100 kernels hipfire is tuned for; native gfx1102 gave the same outputs.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# hipfire unloads this library on exit and then crashes under WSL; keep it resident. The path
# is that of ROCm 7.2.1.
export LD_PRELOAD=/opt/rocm-7.2.1/lib/libamdhip64.so
# shortcut: hipfire 0.4.6's plain Qwen2 route breaks over HTTP (no gen_start), drop --spec once fixed.
exec hipfire serve 127.0.0.1:11435 --no-prewarm --spec ngram
