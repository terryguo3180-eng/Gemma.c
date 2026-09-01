param(
    [Parameter(Mandatory=$true)]
    [string]$File
)

function Set-ContentRetry {
    param(
        [string]$Path,
        [string]$Value,
        [int]$MaxAttempts = 15,
        [int]$DelayMs = 150
    )
    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        try {
            # Use .NET API instead of Set-Content to get clearer exceptions,
            # and avoid Set-Content silently swallowing retry opportunities in the pipeline
            [System.IO.File]::WriteAllText($Path, $Value)
            return
        }
        catch [System.IO.IOException] {
            Start-Sleep -Milliseconds $DelayMs
        }
    }
    throw "Unable to write to $Path (retried $MaxAttempts times, file may be held by another process)"
}

function Get-ContentRetry {
    param(
        [string]$Path,
        [int]$MaxAttempts = 15,
        [int]$DelayMs = 150
    )
    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        try {
            return [System.IO.File]::ReadAllText($Path)
        }
        catch [System.IO.IOException] {
            Start-Sleep -Milliseconds $DelayMs
        }
    }
    throw "Unable to read $Path (retried $MaxAttempts times)"
}

$ErrorActionPreference = 'Stop'

try {
    # Comment out pragmas, and swap `*restrict[ ]+` -> `*const[ ]+__ ` (both 8
    # characters in total). For some reason clang-format can't align `restrict`
    # correctly but `const` works fine.
    $content = Get-ContentRetry -Path $File
    $content = $content -creplace '#pragma omp', '//#pragma omp'
    $content = $content -creplace '\*restrict( +)', '*const$1__ '
    Set-ContentRetry -Path $File -Value $content

    # Run clang-format
    clang-format -i $File
    if ($LASTEXITCODE -ne 0) {
        throw "clang-format returned non-zero exit code: $LASTEXITCODE"
    }

    # After clang-format exits, file handles/antivirus scans sometimes haven't
    # released immediately on Windows; a small wait significantly reduces the
    # chance of lock contention when reading below
    Start-Sleep -Milliseconds 200

    # Restore pragmas and swap `*const[ ]+__ ` back to `*restrict[ ]+`
    $content2 = Get-ContentRetry -Path $File
    $content2 = $content2 -creplace '// ?#pragma omp', '#pragma omp'
    $content2 = $content2 -creplace '\*const( +)__ ', '*restrict$1'
    Set-ContentRetry -Path $File -Value $content2

    Write-Host "OK: $File formatted, pragmas and restrict restored" -ForegroundColor Green
}
catch {
    Write-Host "Failed: $_" -ForegroundColor Red
    Write-Host "File state may be incomplete, please check #pragma omp / restrict in $File for accidental replacement" -ForegroundColor Yellow
    exit 1
}