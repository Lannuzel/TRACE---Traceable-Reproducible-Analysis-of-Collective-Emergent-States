# run_perf_VR_S2.ps1
# ------------------------------------------------------------
# Version "S2 uniquement" (VR - logs Unity type *_ReservationPositionData.csv)
#
# MODIF: ne garde QUE les fichiers situés dans le dossier "modelisateur"
# (donc rôle unique = modelisateur)
#
# - Parcourt tous les dossiers ...\VR\S2\bimXXX
# - Cherche les fichiers participant "*ReservationPositionData*.csv" (ignore *_old.csv)
#   -> uniquement si le fichier est dans ...\bimXXX\modelisateur\...
# - Lance performance_eval.py + eval_consignes_s2.py
# - Écrit les sorties dans $OutRoot\S2\bimXXX\modelisateur\...
# ------------------------------------------------------------

$DataRoot   = "D:\DATA_E2"
$OutRoot    = "D:\Analyse_donnee\Longitudinale\results\performance_task\performance_VR\S2"
$WorkDir    = "D:\Analyse_donnee"

# Scripts (mets ici tes chemins réels)
$PerfScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S2\VR\performance_eval.py"
$ConsScript = ".\Longitudinale\scripts\evaluation_performance_scenario\S2\VR\eval_consignes_s2.py"

# Solution S2 (VR correction)
$SolutionS2 = "D:\Analyse_donnee\Longitudinale\scripts\evaluation_performance_scenario\corrections\VR_S2.csv"

# Paramètres (VR)
$Tolerance  = 0.25
$Axes       = "position.x,position.z"

if (-not (Test-Path $SolutionS2)) {
    Write-Error "Solution S2 introuvable: $SolutionS2"
    exit 2
}

if (-not (Test-Path $PerfScript)) {
    Write-Error "PerfScript introuvable: $PerfScript"
    exit 2
}

if (-not (Test-Path $ConsScript)) {
    Write-Error "ConsScript introuvable: $ConsScript"
    exit 2
}

# 1) Trouver tous les dossiers bimXXX ou bimXXX_2 uniquement dans ...\VR\S2\
$GroupDirs = Get-ChildItem -Path $DataRoot -Directory -Recurse |
    Where-Object {
        $_.Name -match '^bim\d{3}(?:_\d+)?$' -and
        $_.FullName -match '\\VR\\S2\\'
    }

Write-Host ("Found {0} VR/S2 group folders" -f $GroupDirs.Count)


foreach ($g in $GroupDirs) {
    $GroupName = $g.Name

    # 2) Chercher tous les fichiers participant VR UNIQUEMENT dans "modelisateur"
    #    - *ReservationPositionData*.csv
    #    - pas *_old.csv
    #    - filtre chemin: \modelisateur\
    $Candidates = Get-ChildItem -Path $g.FullName -File -Recurse -Filter "*.csv" |
        Where-Object {
            $_.Name -like "*ReservationPositionData*" -and
            $_.Name -notlike "*_old.csv" -and
            $_.FullName -match '\\modelisateur\\'
        } |
        Sort-Object LastWriteTime -Descending

    if (-not $Candidates -or $Candidates.Count -eq 0) {
        Write-Warning ("Skip {0}: no *ReservationPositionData*.csv under modelisateur in {1}" -f $GroupName, $g.FullName)
        continue
    }

    # 2bis) rôle unique = modelisateur (on prend le + récent)
    $role = "modelisateur"
    $Participant = $Candidates | Select-Object -First 1

    Write-Host ("=== S2 | {0} | roles: {1} ===" -f $GroupName, $role)

    Push-Location $WorkDir
    try {
        # 3) Préparer les dossiers de sortie (structure identique)
        $OutBaseDir  = Join-Path $OutRoot "$GroupName"
        $OutPerfDir  = Join-Path $OutBaseDir "performance"
        $OutConsDir  = Join-Path $OutBaseDir "consigne"
        New-Item -ItemType Directory -Force -Path $OutPerfDir, $OutConsDir | Out-Null

        $OutPerfCSV      = Join-Path $OutPerfDir "performance_per_object.csv"
        $OutPerfJSON     = Join-Path $OutPerfDir "performance_summary.json"
        $OutOverlayPerf  = Join-Path $OutPerfDir "overlay.png"

        $OutConsJSON     = Join-Path $OutConsDir "consignes_report.json"
        $OutConsCSV      = Join-Path $OutConsDir "consignes_diag.csv"
        $OutOverlayCons  = Join-Path $OutConsDir "consignes_overlay.png"

        Write-Host ""
        Write-Host ("--- {0} | {1} ---" -f $GroupName, $role)
        Write-Host "Participant: $($Participant.FullName)"
        Write-Host "Solution   : $SolutionS2"

        # --- Commande 1 : performance_eval.py ---
        try {
            python $PerfScript `
                --participant $Participant.FullName `
                --solution $SolutionS2 `
                --axes $Axes `
                --tolerance $Tolerance `
                --pick newest `
                --assignment hungarian `
                --out_csv $OutPerfCSV `
                --out_json $OutPerfJSON `
                --plot $OutOverlayPerf `
        } catch {
            Write-Warning ("Perf failed for {0}/{1}: {2}" -f $GroupName, $role, $_.Exception.Message)
        }

        # --- Commande 2 : eval_consignes_s2.py ---
        try {
            python $ConsScript `
                --participant $Participant.FullName `
                --solution $SolutionS2 `
                --pos_tol     $Tolerance `
                --size_abs    0.01 `
                --size_rel    0.05 `
                --out_json $OutConsJSON `
                --out_csv $OutConsCSV `
                --plot $OutOverlayCons
        } catch {
            Write-Warning ("Consignes failed for {0}/{1}: {2}" -f $GroupName, $role, $_.Exception.Message)
        }
    }
    finally {
        Pop-Location
    }
}
