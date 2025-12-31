# NayukiBlog 初始化脚本 (Windows PowerShell)
# 用于下载 KaTeX 和 Mermaid 本地依赖

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NayukiBlog 初始化脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$libDir = Join-Path $rootDir "frontend/public/lib"
$katexDir = Join-Path $libDir "katex"
$fontsDir = Join-Path $katexDir "fonts"
$mermaidDir = Join-Path $libDir "mermaid"

# 创建目录
Write-Host "[1/5] 创建目录结构..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $fontsDir | Out-Null
New-Item -ItemType Directory -Force -Path $mermaidDir | Out-Null
Write-Host "  ✓ 目录创建完成" -ForegroundColor Green

# 下载 KaTeX 核心文件
Write-Host "[2/5] 下载 KaTeX 核心文件..." -ForegroundColor Yellow
$katexFiles = @(
    @{ url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"; name = "katex.min.css" },
    @{ url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"; name = "katex.min.js" },
    @{ url = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"; name = "auto-render.min.js" }
)

foreach ($file in $katexFiles) {
    $outPath = Join-Path $katexDir $file.name
    Write-Host "  下载 $($file.name)..." -NoNewline
    Invoke-WebRequest -Uri $file.url -OutFile $outPath -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
}

# 下载 KaTeX 字体
Write-Host "[3/5] 下载 KaTeX 字体文件..." -ForegroundColor Yellow
$fonts = @(
    "KaTeX_AMS-Regular.woff2",
    "KaTeX_Caligraphic-Bold.woff2",
    "KaTeX_Caligraphic-Regular.woff2",
    "KaTeX_Fraktur-Bold.woff2",
    "KaTeX_Fraktur-Regular.woff2",
    "KaTeX_Main-Bold.woff2",
    "KaTeX_Main-BoldItalic.woff2",
    "KaTeX_Main-Italic.woff2",
    "KaTeX_Main-Regular.woff2",
    "KaTeX_Math-BoldItalic.woff2",
    "KaTeX_Math-Italic.woff2",
    "KaTeX_SansSerif-Bold.woff2",
    "KaTeX_SansSerif-Italic.woff2",
    "KaTeX_SansSerif-Regular.woff2",
    "KaTeX_Script-Regular.woff2",
    "KaTeX_Size1-Regular.woff2",
    "KaTeX_Size2-Regular.woff2",
    "KaTeX_Size3-Regular.woff2",
    "KaTeX_Size4-Regular.woff2",
    "KaTeX_Typewriter-Regular.woff2"
)

$fontCount = 0
foreach ($font in $fonts) {
    $fontCount++
    $outPath = Join-Path $fontsDir $font
    Write-Host "  [$fontCount/$($fonts.Count)] $font..." -NoNewline
    Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/$font" -OutFile $outPath -UseBasicParsing
    Write-Host " ✓" -ForegroundColor Green
}

# 下载 Mermaid
Write-Host "[4/5] 下载 Mermaid..." -ForegroundColor Yellow
$mermaidPath = Join-Path $mermaidDir "mermaid.min.js"
Write-Host "  下载 mermaid.min.js..." -NoNewline
Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js" -OutFile $mermaidPath -UseBasicParsing
Write-Host " ✓" -ForegroundColor Green

# 修改 CSS 中的字体路径
Write-Host "[5/5] 修改 CSS 字体路径..." -ForegroundColor Yellow
$cssPath = Join-Path $katexDir "katex.min.css"
$css = Get-Content $cssPath -Raw
$css = $css -replace 'url\(fonts/', 'url(/lib/katex/fonts/'
Set-Content $cssPath $css -NoNewline
Write-Host "  ✓ CSS 路径修改完成" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  初始化完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "  cd frontend && npm install && npm run dev"
