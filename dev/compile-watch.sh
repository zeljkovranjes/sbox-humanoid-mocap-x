#!/usr/bin/env bash
# Prints every s&box editor compile failure the moment any running editor logs it.
# `dotnet build dev/Editor.csproj` misses engine-only errors (code generation, whitelist),
# so watch the engine's own logs. Usage: dev/compile-watch.sh [engine logs directory]
set -u
logs=${1:-${FACEPUNCH_ENGINE:-/mnt/projects/sbox/engine/game}/logs}
[ -d "$logs" ] || { echo "No engine log directory at $logs" >&2; exit 2; }
declare -A seen
for file in "$logs"/*.log; do seen[$file]=$(wc -l < "$file"); done
echo "Watching $logs for compile errors..."
while true; do
	for file in "$logs"/*.log; do
		lines=$(wc -l < "$file" 2>/dev/null || echo 0)
		# A file that appears later is the previous log being rotated; skip what it already holds.
		if [ -z "${seen[$file]+set}" ]; then seen[$file]=$lines; continue; fi
		old=${seen[$file]}
		if [ "$lines" -lt "$old" ]; then old=0; fi
		if [ "$lines" -gt "$old" ]; then
			tail -n +$((old + 1)) "$file" | head -n $((lines - old)) \
				| grep -E "Compile of .* Failed|Error \||Whitelist violation|Broken Reference" \
				| sed "s|^|$(basename "$file"): |"
		fi
		seen[$file]=$lines
	done
	sleep 2
done
