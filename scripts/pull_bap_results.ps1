# Pull Tier-4 Bayes-Phi results from the VM into THIS clone.
# The VM run (python src/tier4_bayes.py) writes output/tier4_bayes_bundle.tar.gz with
# repo-root-relative paths, so extracting at the local repo root restores
# output/tier4_bayes/* (CSVs, weights, log, curve) and artifacts/tier4_probe.pt in place.
#
# Usage (from anywhere inside the repo):
#   powershell scripts/pull_tier4_bayes_results.ps1
#   powershell scripts/pull_tier4_bayes_results.ps1 -VmHost other-vm.example.com -Port 22

param(
    [string]$VmHost = "lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com",
    [string]$User = "disi",
    [int]$Port = 5040,
    [string]$RemoteRepo = "~/Deep-Learning-Project"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder — works in any clone.
$repoRoot = Split-Path -Parent $PSScriptRoot
$bundleName = "tier4_bayes_bundle.tar.gz"
$localBundle = Join-Path $repoRoot $bundleName

Write-Host "Fetching $bundleName from ${User}@${VmHost}:${Port} ..."
scp -P $Port "${User}@${VmHost}:${RemoteRepo}/output/${bundleName}" $localBundle
if (-not (Test-Path $localBundle)) { throw "scp did not produce $localBundle" }

Write-Host "Extracting into $repoRoot ..."
tar -xzf $localBundle -C $repoRoot
Remove-Item $localBundle -Confirm:$false

Write-Host "Done. Results restored under:"
Write-Host "  $repoRoot\output\tier4_bayes\   (CSVs, weights, train_log.txt, training curve)"
Write-Host "  $repoRoot\artifacts\tier4_probe.pt"
Get-ChildItem (Join-Path $repoRoot "output\tier4_bayes") | Select-Object Name, Length | Format-Table -AutoSize
