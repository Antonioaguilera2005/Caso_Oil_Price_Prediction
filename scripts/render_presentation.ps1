param(
    [Parameter(Mandatory=$true)][string]$PresentationPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$PdfPath = ''
)
$ErrorActionPreference = 'Stop'
$taskInput = (Resolve-Path -LiteralPath $PresentationPath).Path
$taskOutput = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($taskOutput) | Out-Null
$taskPowerPoint = $null
$taskPresentation = $null
try {
    $taskPowerPoint = New-Object -ComObject PowerPoint.Application
    $taskInitialCount = $taskPowerPoint.Presentations.Count
    $taskPowerPoint.DisplayAlerts = 1
    $taskPresentation = $taskPowerPoint.Presentations.Open($taskInput, -1, 0, 0)
    $taskTextBounds = @()
    foreach ($taskSlide in $taskPresentation.Slides) {
        $taskSlidePath = Join-Path $taskOutput ('slide-{0:D2}.png' -f $taskSlide.SlideIndex)
        $taskSlide.Export($taskSlidePath, 'PNG', 1920, 1080)
        foreach ($taskShape in $taskSlide.Shapes) {
            if ($taskShape.Name -like 'PRESENTACION_*' -and $taskShape.HasTextFrame -eq -1 -and $taskShape.TextFrame2.HasText -eq -1) {
                $taskTextBounds += [pscustomobject]@{
                    slide=$taskSlide.SlideIndex; name=$taskShape.Name
                    text=$taskShape.TextFrame2.TextRange.Text
                    boxHeight=[math]::Round($taskShape.Height,2)
                    boundHeight=[math]::Round($taskShape.TextFrame2.TextRange.BoundHeight,2)
                    boxWidth=[math]::Round($taskShape.Width,2)
                    boundWidth=[math]::Round($taskShape.TextFrame2.TextRange.BoundWidth,2)
                    overflow=($taskShape.TextFrame2.TextRange.BoundHeight -gt $taskShape.Height + 1 -or $taskShape.TextFrame2.TextRange.BoundWidth -gt $taskShape.Width + 1)
                }
            }
        }
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskSlide) | Out-Null
    }
    $taskTextBounds | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskOutput 'text-bounds.json') -Encoding UTF8
    if ($PdfPath) {
        $taskPdf = [IO.Path]::GetFullPath($PdfPath)
        $taskPresentation.SaveAs($taskPdf, 32)
    }
    Write-Output ('Rendered {0} slides.' -f $taskPresentation.Slides.Count)
}
finally {
    if ($null -ne $taskPresentation) {
        $taskPresentation.Close()
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskPresentation) | Out-Null
    }
    if ($null -ne $taskPowerPoint) {
        if ($taskInitialCount -eq 0 -and $taskPowerPoint.Presentations.Count -eq 0) { $taskPowerPoint.Quit() }
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskPowerPoint) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
