# Pull the Tier-3 DualScore multi-seed results (CSVs, weights, per-source dumps) from the VM.
# VM run that produces them (from the VM repo root):
#   python src/tier3_dualscore.py --seeds 0,1,2 --dump-per-source
#
# Usage (from anywhere inside the repo):
#   powershell scripts/pull_tier3_dualscore_seeds.ps1

param(
    [string]$VmHost = "lab-d2fdded3-9ba4-4188-80fa-16c27a4cd022.westeurope.cloudapp.azure.com",
    [string]$User = "disi",
    [int]$Port = 5037,
    [string]$RemoteRepo = "~/DLproject_VM/Deep-Learning-Project"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

foreach ($sub in @("output/tier3_dualscore", "output/per_source")) {
    $local = Join-Path $repoRoot ($sub -replace "/", "\")
    New-Item -ItemType Directory -Force $local | Out-Null
    Write-Host "Fetching ${RemoteRepo}/${sub}/* ..."
    scp -P $Port "${User}@${VmHost}:${RemoteRepo}/${sub}/tier3_dualscore_seed*" $local
}

Write-Host "Done. Seed CSVs/weights in output\tier3_dualscore\, dumps in output\per_source\"
Get-ChildItem (Join-Path $repoRoot "output\per_source") | Select-Object Name, Length | Format-Table -AutoSize
