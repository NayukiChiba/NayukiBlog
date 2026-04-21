---
title: Vscode中的Latex设置
date: 2026-04-21
category: 技术
tags:
  - 工具
  - 设置
description: 在vscode中使用latexworkshop的时候，settings.json设置Latex
image: https://img.yumeko.site/file/blog/Latex.jpg
status: published
---
# Settings.json配置Latex
```text
//------------------------------LaTeX 配置----------------------------------
  // 设置是否自动编译
  "latex-workshop.latex.autoBuild.run": "never",
  //右键菜单
  "latex-workshop.showContextMenu": true,
  //从使用的包中自动补全命令和环境
  "latex-workshop.intellisense.package.enabled": true,
  //编译出错时设置是否弹出气泡设置
  "latex-workshop.message.error.show": false,
  "latex-workshop.message.warning.show": false,
  // latexmk生成的pdf文件输出目录，%DIR%为.tex文件所在目录，%DOCFILE%为.tex文件名（不带扩展名）
  "latex-workshop.latex.outDir": "%DIR%/build",
  "latex-workshop.latex.tools": [
    {
      "name": "xelatex",
      "command": "xelatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-output-directory=%DIR%/build",
        "%DOCFILE%",
      ],
    },
    {
      "name": "pdflatex",
      "command": "pdflatex",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-output-directory=%DIR%/build",
        "%DOCFILE%",
      ],
    },
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-pdf",
        "-outdir=%DIR%/build",
        "%DOCFILE%",
      ],
    },
    {
      "name": "bibtex",
      "command": "bibtex",
      "args": ["%DIR%/build/%DOCFILE%"],
    },
  ],
  // 用于配置编译链
  "latex-workshop.latex.recipes": [
    {
      "name": "XeLaTeX",
      "tools": ["xelatex"],
    },
    {
      "name": "PDFLaTeX",
      "tools": ["pdflatex"],
    },
    {
      "name": "BibTeX",
      "tools": ["bibtex"],
    },
    {
      "name": "LaTeXmk",
      "tools": ["latexmk"],
    },
    {
      "name": "xelatex -> bibtex -> xelatex*2",
      "tools": ["xelatex", "bibtex", "xelatex", "xelatex"],
    },
    {
      "name": "pdflatex -> bibtex -> pdflatex*2",
      "tools": ["pdflatex", "bibtex", "pdflatex", "pdflatex"],
    },
  ],
  //文件清理。此属性必须是字符串数组
  "latex-workshop.latex.clean.fileTypes": [
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.idx",
    "*.ind",
    "*.lof",
    "*.lot",
    "*.out",
    "*.toc",
    "*.acn",
    "*.acr",
    "*.alg",
    "*.glg",
    "*.glo",
    "*.gls",
    "*.ist",
    "*.fls",
    "*.log",
    "*.fdb_latexmk",
  ],
  //设置为onFaild 在构建失败后清除辅助文件
  "latex-workshop.latex.autoClean.run": "onFailed",
  // 使用上次的recipe编译组合
  "latex-workshop.latex.recipe.default": "lastUsed",
  // 用于反向同步的内部查看器的键绑定。ctrl/cmd +点击(默认)或双击
  "latex-workshop.view.pdf.internal.synctex.keybinding": "double-click",
  // 设置VScode内部查看生成的pdf文件
  "latex-workshop.view.pdf.viewer": "browser",
```