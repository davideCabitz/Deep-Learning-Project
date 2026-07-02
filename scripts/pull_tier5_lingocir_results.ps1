# Pull Tier-5 LingoCIR results from the VM into THIS clone.
# The VM run (python src/tier5_lingocir.py) writes output/tier5_lingocir_bundle.tar.gz with
# repo-root-relative paths, so extracting at the local repo root restores
# output/tier5_lingocir/* (CSVs, log, curve) and artifacts/tier5_lingocir_g.pt in place.
#
# Usage (from anywhere inside the repo):
#   powershell scripts/pull_tier5_lingocir_results.ps1
#   powershell scripts/pull_tier5_lingocir_results.ps1 -VmHost other-vm.example.com -Port 22

param(
    [string]$VmHost = "lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com",
    [string]$User = "disi",
    [int]$Port = 5040,
    [string]$RemoteRepo = "~/Deep-Learning-Project"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder — works in any clone.
$repoRoot = Split-Path -Parent $PSScriptRoot
$bundleName = "tier5_lingocir_bundle.tar.gz"
$localBundle = Join-Path $repoRoot $bundleName

Write-Host "Fetching $bundleName from ${User}@${VmHost}:${Port} ..."
scp -P $Port "${User}@${VmHost}:${RemoteRepo}/output/${bundleName}" $localBundle
if (-not (Test-Path $localBundle)) { throw "scp did not produce $localBundle" }

Write-Host "Extracting into $repoRoot ..."
tar -xzf $localBundle -C $repoRoot
Remove-Item $localBundle -Confirm:$false

Write-Host "Done. Results restored under:"
Write-Host "  $repoRoot\output\tier5_lingocir\   (CSVs, train_log.txt, training curve)"
Write-Host "  $repoRoot\artifacts\tier5_lingocir_g.pt"
Get-ChildItem (Join-Path $repoRoot "output\tier5_lingocir") | Select-Object Name, Length | Format-Table -AutoSize
