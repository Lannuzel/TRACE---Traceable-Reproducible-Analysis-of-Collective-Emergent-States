# run_perf_pc_revit_s1.ps1
# ------------------------------------------------------------
# Version "S1 uniquement" (PC, exports Revit déjà convertis en PC-like)
#
# - Parcourt tous les dossiers ...\PC\S1\bimXXX
# - Cherche le CSV participant dans bimXXX (ignore *_old.csv) et exige "PositionX" dans l’entête
# - Lance performance_eval.py + eval_consignes_align.py
# - Écrit les sorties dans $OutRoot\S1\bimXXX\...
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
$OutRoot    = "D:\Analyse_donnee\Longitudinale\results\performance_task\performance_PC\S1"
$WorkDir    = "D:\Analyse_donnee"

# Scripts (mets ici tes chemins réels côté PC)
# $PerfScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S1\PC\performance_eval.py"
$PerfScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S1\performance_eval_unified.py"
$ConsScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S1\PC\eval_consignes_align.py"

# Solution S1 (CSV ou XLSX selon ton script)
$SolutionS1 = "D:\Analyse_donnee\Longitudinale\scripts\evaluation_performance_scenario\corrections\PC_s1.csv"

if (-not (Test-Path $SolutionS1)) {
    Write-Error "Solution S1 introuvable: $SolutionS1"
    exit 2
}

# 1) Trouver tous les dossiers bimXXX uniquement dans ...\PC\S1\
$GroupDirs = Get-ChildItem -Path $DataRoot -Directory -Recurse |
    Where-Object { $_.Name -match '^bim\d{3}$' -and $_.FullName -match '\\PC\\S1\\' }

Write-Host ("Found {0} PC/S1 group folders" -f $GroupDirs.Count)

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
        if ($header -match '(^|;)PositionX(;|$)') {
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

    Write-Host "=== S1 | $GroupName ==="
    Write-Host "Participant: $($Participant.FullName)"
    Write-Host "Solution   : $SolutionS1"

    Push-Location $WorkDir
    try {
        # --- Commande 1 : performance_eval.py ---
        try {
            # python $PerfScript `
            #     --participant $Participant.FullName `
            #     --solution $SolutionS1 `
            #     --tolerance $Tolerance `
            #     --axes $Axes `
            #     --out_csv $OutPerfCSV `
            #     --out_json $OutPerfJSON `
            #     --plot $OutOverlayPerf
            python $PerfScript `
                --participant $Participant.FullName `
                --solution $SolutionS1 `
                --solution_space revit `
                --participant_space revit `
                --calib translation `
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
