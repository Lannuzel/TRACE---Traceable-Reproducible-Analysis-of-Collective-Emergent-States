# run_perf_s2_PC.ps1
# ------------------------------------------------------------
# Version "S2 uniquement" (PC, exports Revit déjà convertis en PC-like)
#
# - Parcourt tous les dossiers ...\PC\S2\bimXXX
# - Cherche le CSV participant dans bimXXX (ignore *_old.csv) et exige "PositionX" dans l’entête
# - Lance performance_eval.py + eval_consignes_s2.py
# - Écrit les sorties dans $OutRoot\S2\bimXXX\...
#
# A MODIFIER:
# - $DataRoot
# - $WorkDir
# - $PerfScript / $ConsScript
# - $SolutionS1
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_perf_pc_revit_s1.ps1
# ------------------------------------------------------------

$DataRoot   = "D:\DATA_E2"
$OutRoot    = "D:\Analyse_donnee\Longitudinale\results\performance_task\performance_PC\S2"
$WorkDir    = "D:\Analyse_donnee"

# Scripts (mets ici tes chemins réels côté PC)
$PerfScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S2\PC\performance_eval.py"
$ConsScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S2\PC\eval_consignes_s2.py"

# Solution S2 (CSV ou XLSX selon ton script)
$SolutionS1 = "D:\Analyse_donnee\Longitudinale\scripts\evaluation_performance_scenario\corrections\PC_s2.csv"

# Paramètres
$Tolerance  = 0.25     # si besoin Revit: 0.30-0.40
$Axes       = "CenterX_m,CenterY_m"

if (-not (Test-Path $SolutionS1)) {
    Write-Error "Solution S2 introuvable: $SolutionS1"
    exit 2
}

# 1) Trouver tous les dossiers bimXXX uniquement dans ...\PC\S2\
$GroupDirs = Get-ChildItem -Path $DataRoot -Directory -Recurse |
    Where-Object { $_.Name -match '^bim\d{3}$' -and $_.FullName -match '\\PC\\S2\\' }

Write-Host ("Found {0} PC/S2 group folders" -f $GroupDirs.Count)

foreach ($g in $GroupDirs) {
    $GroupName = $g.Name

    # 2) Trouver le fichier participant dans le dossier bimXXX :
    #    - *.csv
    #    - pas *_old.csv
    #    - contient "PositionX" dans l'entête
    $Candidates = Get-ChildItem -Path $g.FullName -File -Filter "*.csv" |
        Where-Object { $_.Name -notlike "*_old.csv" } |
        Sort-Object LastWriteTime -Descending

    $Participant = $null
    foreach ($c in $Candidates) {
        if ($c.Length -le 10) { continue }
        $header = Get-Content -LiteralPath $c.FullName -TotalCount 1 -ErrorAction SilentlyContinue
        if ($null -eq $header) { continue }
        if ($header -match '(^|;)CenterX_m(;|$)') {
            $Participant = $c
            break
        }
    }

    if (-not $Participant) {
        Write-Warning ("Skip {0}: no valid PC-like CSV (PositionX) in {1}" -f $GroupName, $g.FullName)
        continue
    }

    # 3) Préparer les dossiers de sortie
    $OutPerfDir = Join-Path $OutRoot "$GroupName\performance"
    $OutConsDir = Join-Path $OutRoot "$GroupName\consigne"
    New-Item -ItemType Directory -Force -Path $OutPerfDir, $OutConsDir | Out-Null

    $OutPerfCSV      = Join-Path $OutPerfDir "performance_per_object.csv"
    $OutPerfJSON     = Join-Path $OutPerfDir "performance_summary.json"
    $OutOverlayPerf  = Join-Path $OutPerfDir "overlay.png"

    $OutConsJSON     = Join-Path $OutConsDir "consignes_report.json"
    $OutConsCSV      = Join-Path $OutConsDir "consignes_diag.csv"
    $OutOverlayCons  = Join-Path $OutConsDir "consignes_overlay.png"

    Write-Host "=== S2 | $GroupName ==="
    Write-Host "Participant: $($Participant.FullName)"
    Write-Host "Solution   : $SolutionS1"

    Push-Location $WorkDir
    try {
        # --- Commande 1 : performance_eval.py ---
        try {
            python $PerfScript `
                --participant $Participant.FullName `
                --solution $SolutionS1 `
                --axes $Axes `
                --tolerance $Tolerance `
                --calib none `
                --out_csv $OutPerfCSV `
                --out_json $OutPerfJSON `
                --plot $OutOverlayPerf
        } catch {
            Write-Warning ("Perf failed for {0}: {1}" -f $GroupName, $_.Exception.Message)
        }

        # --- Commande 2 : eval_consignes_align.py ---
        try {
            python $ConsScript `
                --participant $Participant.FullName `
                --solution $SolutionS1 `
                --pos_tol     $Tolerance `
                --size_abs    0.01 `
                --size_rel    0.05 `
                --out_json $OutConsJSON `
                --out_csv $OutConsCSV `
                --plot $OutOverlayCons
        } catch {
            Write-Warning ("Consignes failed for {0}: {1}" -f $GroupName, $_.Exception.Message)
        }
    }
    finally {
        Pop-Location
    }
}
