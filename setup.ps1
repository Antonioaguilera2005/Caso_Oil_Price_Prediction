$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.tools\python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $ProjectRoot '.tools\bin'
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot '.cache\uv'
    $env:POETRY_CACHE_DIR = Join-Path $ProjectRoot '.cache\pypoetry'
    $env:POETRY_VIRTUALENVS_IN_PROJECT = 'true'
    New-Item -ItemType Directory -Force '.tools' | Out-Null
    $UvExecutable = Join-Path $ProjectRoot '.tools\uv\uv.exe'
    if (-not (Test-Path -LiteralPath $UvExecutable)) {
        Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\uv.zip'
        Expand-Archive -LiteralPath '.tools\uv.zip' -DestinationPath '.tools\uv' -Force
    }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $UvExecutable python install 3.12
        if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar Python' }
        & $UvExecutable venv .venv --python 3.12
        if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno' }
    }
    & $UvExecutable pip install --python '.venv\Scripts\python.exe' 'poetry==2.5.1'
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar Poetry' }
    $env:VIRTUAL_ENV = Join-Path $ProjectRoot '.venv'
    & '.\.venv\Scripts\python.exe' -m poetry install --no-interaction
    if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias' }
    & '.\.venv\Scripts\python.exe' -m scripts.build_app
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo construir la aplicacion HTML' }
    Write-Output 'Listo. Ejecuta: .\.venv\Scripts\python.exe -m streamlit run app.py'
}
finally {
    Pop-Location
}
