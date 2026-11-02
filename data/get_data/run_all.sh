#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
runner="$script_dir/$(basename -- "${BASH_SOURCE[0]}")"

export HF_HOME="${HF_HOME:-/work/c8763c8763/huggingface-cache}"
mkdir -p "$HF_HOME"

cd "$repo_root"

for script in "$script_dir"/*.sh; do
	[[ -f "$script" ]] || continue
	[[ "$script" == "$runner" ]] && continue

	printf '\n=== Running %s ===\n' "$(basename -- "$script")"
	bash "$script"
done

printf '\nAll data download scripts completed.\n'
