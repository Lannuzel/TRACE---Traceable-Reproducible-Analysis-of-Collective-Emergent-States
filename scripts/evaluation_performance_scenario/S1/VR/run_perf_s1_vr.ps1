# run_perf_s1_vr.ps1
# Exécute performance_eval.py + eval_consignes_align.py pour tous les groupes bimXXX sous D:\DATA_E2\**\VR\S1\

$DataRoot    = "D:\DATA_E2"
$OutRoot     = "D:\Analyse_donnee\Longitudinale\results\performance_task\performance_VR\S1"
$WorkDir     = "D:\Analyse_donnee"  # dossier où tu lances les python .\Longitudinale\...

$PerfScript  = ".\Longitudinale\scripts\evaluation_performance_scenario\S1\performance_eval_unified.py"
$ConsScript  = ".\Longitudinale\scripts\evaluation_performance_scenario\S1\VR\eval_consignes_align.py"

$SolutionXLSX = "D:\Analyse_donnee\Longitudinale\scripts\evaluation_performance_scenario\corrections\VR_S1.xlsx"

# 1) Trouver tous les dossiers bimXXX ou bimXXX_2 uniquement dans ...\VR\S1\
$GroupDirs = Get-ChildItem -Path $DataRoot -Directory -Recurse |
    Where-Object {
        $_.Name -match '^bim\d{3}(?:_\d+)?$' -and
        $_.FullName -match '\\VR\\S1\\'
    }

Write-Host ("Found {0} VR/S1 group folders" -f $GroupDirs.Count)


Write-Host "Groupes trouvés :"
$GroupDirs | ForEach-Object { Write-Host $_.FullName }

foreach ($g in $GroupDirs) {
    $GroupName = $g.Name
    $ModelDir  = Join-Path $g.FullName "modelisateur"

    if (-not (Test-Path $ModelDir)) {
        Write-Warning "Skip ${GroupName}: no modelisateur folder ($ModelDir)"        
        continue
    }

    # # 2) Trouver le fichier participant: *_ChairePositionData.csv dans modelisateur\
    # $Participant = Get-ChildItem -Path $ModelDir -File -Filter "*_ChairePositionData.csv" |
    #                Select-Object -First 1

    # Candidats (plus récents d'abord)
    $Candidates = Get-ChildItem -Path $ModelDir -File -Filter "*_ChairePositionData.csv" |
                Sort-Object LastWriteTime -Descending

    # On garde le premier fichier:
    # - pas quasi vide
    # - contenant "PositionX" (header attendu)
   $Participant = $Candidates | Where-Object { $_.Length -gt 10 } | Select-Object -First 1
    if (-not $Participant) {
        Write-Warning ("Skip {0}: only empty *_ChairePositionData.csv in {1}" -f $GroupName, $ModelDir)
        continue
    }


    # 3) Préparer les dossiers de sortie
    $OutPerfDir = Join-Path $OutRoot "$GroupName\performance"
    $OutConsDir = Join-Path $OutRoot "$GroupName\consigne"
    New-Item -ItemType Directory -Force -Path $OutPerfDir, $OutConsDir | Out-Null

    $OutPerfCSV  = Join-Path $OutPerfDir "performance_per_object.csv"
    $OutPerfJSON = Join-Path $OutPerfDir "performance_summary.json"
    $OutOverlayPerf = Join-Path $OutPerfDir "overlay.png"

    $OutConsJSON = Join-Path $OutConsDir "consignes_report.json"
    $OutConsCSV  = Join-Path $OutConsDir "consignes_diag.csv"     # <- je corrige ici: un CSV par groupe (sinon écrasement)
    $OutOverlayCons = Join-Path $OutConsDir "consignes_overlay.png"

    Write-Host "=== $GroupName ==="
    Write-Host "Participant: $($Participant.FullName)"

    Push-Location $WorkDir
    try {
        try {
        # --- Commande 1 : performance_eval.py ---
        # python $PerfScript `
        #     --participant $Participant.FullName `
        #     --solution $SolutionXLSX `
        #     --tolerance 0.25 `
        #     --axes "PositionX,PositionZ" `
        #     --out_csv $OutPerfCSV `
        #     --out_json $OutPerfJSON `
        #     --plot $OutOverlayPerf
        python $PerfScript `
            --participant $Participant.FullName `
            --solution $SolutionXLSX `
            --solution_space unity `
            --participant_space unity `
            --calib translation `
            --out_csv $OutPerfCSV `
            --out_json $OutPerfJSON `
            --plot $OutOverlayPerf
        } catch {
            Write-Warning ("Perf failed for {0}: {1}" -f $GroupName, $_.Exception.Message)
        }
        try{
        # --- Commande 2 : eval_consignes_align.py ---
        python $ConsScript `
            --participant $Participant.FullName `
            --solution $SolutionXLSX `
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
