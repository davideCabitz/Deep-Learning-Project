# Pull Tier-5 CD-Bayes results from the VM into THIS clone.
# The VM run (python src/tier5_cdbayes.py) writes output/tier5_cdbayes_bundle.tar.gz with
# repo-root-relative paths, so extracting at the local repo root restores
# output/tier5_cdbayes/* (ablation CSVs, Phi weights, curves, log) plus
# artifacts/tier4_probe.pt, artifacts/tier5_temperatures.pt and artifacts/tier5_chowliu.pt.
#
# Usage (from anywhere inside the repo):
#   powershell scripts/pull_tier5_cdbayes_results.ps1
#   powershell scripts/pull_tier5_cdbayes_results.ps1 -VmHost other-vm.example.com -Port 22

param(
    [string]$VmHost = "lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com",
    [string]$User = "disi",
    [int]$Port = 5037,
    [string]$RemoteRepo = "~/DLproject_VM/Deep-Learning-Project"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder — works in any clone.
$repoRoot = Split-Path -Parent $PSScriptRoot
$bundleName = "tier5_cdbayes_bundle.tar.gz"
$localBundle = Join-Path $repoRoot $bundleName

Write-Host "Fetching $bundleName from ${User}@${VmHost}:${Port} ..."
scp -P $Port "${User}@${VmHost}:${RemoteRepo}/output/${bundleName}" $localBundle
if (-not (Test-Path $localBundle)) { throw "scp did not produce $localBundle" }

Write-Host "Extracting into $repoRoot ..."
tar -xzf $localBundle -C $repoRoot
Remove-Item $localBundle -Confirm:$false

Write-Host "Done. Results restored under:"
Write-Host "  $repoRoot\output\tier5_cdbayes\   (ablation CSVs, weights, curves, train_log.txt)"
Write-Host "  $repoRoot\artifacts\tier4_probe.pt, tier5_temperatures.pt, tier5_chowliu.pt"
Get-ChildItem (Join-Path $repoRoot "output\tier5_cdbayes") | Select-Object Name, Length | Format-Table -AutoSize
