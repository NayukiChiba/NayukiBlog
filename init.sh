#!/bin/bash
# NayukiBlog 初始化脚本 (Linux/macOS)
# 用于下载 KaTeX 和 Mermaid 本地依赖

set -e

echo "========================================"
echo "  NayukiBlog 初始化脚本"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/frontend/public/lib"
KATEX_DIR="$LIB_DIR/katex"
FONTS_DIR="$KATEX_DIR/fonts"
MERMAID_DIR="$LIB_DIR/mermaid"

# 创建目录
echo "[1/4] 创建目录结构..."
mkdir -p "$FONTS_DIR"
mkdir -p "$MERMAID_DIR"
echo "  ✓ 目录创建完成"

# 下载 KaTeX 核心文件
echo "[2/4] 下载 KaTeX 核心文件..."
cd "$KATEX_DIR"

download_if_not_exists() {
    local url=$1
    local filename=$2
    if [ -f "$filename" ]; then
        echo "  跳过 $filename (已存在)"
    else
        echo -n "  下载 $filename..."
        curl -sO "$url"
        echo " ✓"
    fi
}

download_if_not_exists "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" "katex.min.css"
download_if_not_exists "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" "katex.min.js"
download_if_not_exists "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" "auto-render.min.js"

# 下载 KaTeX 字体
echo "[3/4] 下载 KaTeX 字体文件..."
cd "$FONTS_DIR"
fonts=(
    "KaTeX_AMS-Regular.woff2"
    "KaTeX_Caligraphic-Bold.woff2"
    "KaTeX_Caligraphic-Regular.woff2"
    "KaTeX_Fraktur-Bold.woff2"
    "KaTeX_Fraktur-Regular.woff2"
    "KaTeX_Main-Bold.woff2"
    "KaTeX_Main-BoldItalic.woff2"
    "KaTeX_Main-Italic.woff2"
    "KaTeX_Main-Regular.woff2"
    "KaTeX_Math-BoldItalic.woff2"
    "KaTeX_Math-Italic.woff2"
    "KaTeX_SansSerif-Bold.woff2"
    "KaTeX_SansSerif-Italic.woff2"
    "KaTeX_SansSerif-Regular.woff2"
    "KaTeX_Script-Regular.woff2"
    "KaTeX_Size1-Regular.woff2"
    "KaTeX_Size2-Regular.woff2"
    "KaTeX_Size3-Regular.woff2"
    "KaTeX_Size4-Regular.woff2"
    "KaTeX_Typewriter-Regular.woff2"
)

count=0
skipped=0
total=${#fonts[@]}
for font in "${fonts[@]}"; do
    count=$((count + 1))
    if [ -f "$font" ]; then
        skipped=$((skipped + 1))
    else
        echo -n "  [$count/$total] $font..."
        curl -sO "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/$font"
        echo " ✓"
    fi
done
if [ $skipped -gt 0 ]; then
    echo "  跳过 $skipped 个已存在的字体文件"
fi

# 下载 Mermaid
echo "[4/4] 下载 Mermaid..."
cd "$MERMAID_DIR"
if [ -f "mermaid.min.js" ]; then
    echo "  跳过 mermaid.min.js (已存在)"
else
    echo -n "  下载 mermaid.min.js..."
    curl -sO "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"
    echo " ✓"
fi

echo ""
echo "========================================"
echo "  初始化完成！"
echo "========================================"
echo ""
echo "下一步："
echo "  cd frontend && npm install && npm run dev"
