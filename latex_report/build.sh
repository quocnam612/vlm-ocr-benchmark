#!/bin/sh
set -eu
cd "$(dirname "$0")"
TZ=Asia/Ho_Chi_Minh latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
