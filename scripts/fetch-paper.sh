#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p research/sources
curl --fail --location https://arxiv.org/pdf/2609.05275v1 -o research/sources/2609.05275.pdf
pdftotext -layout research/sources/2609.05275.pdf research/sources/2609.05275.txt
