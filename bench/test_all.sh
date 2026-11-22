#!/bin/zsh
# Every check of the Chippy pool in one go (m5, Chippy on :11445). Prints PASS/FAIL per check; rc = failures.
# Sends embed load: run on a quiet-ish m5, not while another ANE bench runs.
cd "${0:A:h}/.."
P=http://127.0.0.1:11445; F=0
ok() { echo "PASS  $1"; }; bad() { echo "FAIL  $1"; F=$((F+1)); }
check() { local name=$1; shift; if eval "$@" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi; }
echo "== unit tests (swift test)"
(cd app && PATH="$HOME/.cargo/bin:$PATH" swift test 2>&1 | tail -2)
[[ ${pipestatus[1]} -eq 0 ]] && ok "swift test" || bad "swift test"
echo "== server"
check "health answers, CalVer version"   "curl -fsS $P/health | python3 -c 'import sys,json,re; h=json.load(sys.stdin); assert re.fullmatch(r\"\\d{2}\\.\\d{1,2}\\.\\d{1,2}\", h[\"app_version\"]), h[\"app_version\"]'"
check "bad pool value -> 400"            "[[ \$(curl -s -o /dev/null -w '%{http_code}' -XPOST $P/api/embed -d '{\"input\":[\"a\",\"b\"],\"pool\":\"x\"}') == 400 ]]"
check "solo request -> 200"              "[[ \$(curl -s -o /dev/null -w '%{http_code}' -XPOST $P/api/embed -d '{\"input\":[\"a\",\"b\"],\"pool\":\"solo\"}') == 200 ]]"
check "claim with wrong token -> 401"    "[[ \$(curl -s -o /dev/null -w '%{http_code}' -XPOST $P/api/work/claim -d '{\"token\":\"nope\",\"worker\":\"t\",\"identity\":\"x\",\"wait\":1}') == 401 ]]"
LAN=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)
check "LAN listener refuses /api/embed (403)" "[[ \$(curl -s -o /dev/null -w '%{http_code}' -XPOST http://$LAN:11445/api/embed -d '{\"input\":[\"a\"]}') == 403 ]]"
check "LAN listener serves join.sh kit"  "curl -fsS 'http://$LAN:11445/join.sh?t='\$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser(\"~/Library/Application Support/ANEEmbed/join.json\")))[\"token\"])') | head -1 | rg -q '^#!'"
echo "== volunteers: $(curl -s $P/api/workers | rg -o '"id":"pull:[^"]+"' | tr '\n' ' ')"
echo "== cheater"
python3 bench/cheater_test.py | tail -1; [[ ${pipestatus[1]} -eq 0 ]] && ok "cheater cut off, answers clean" || bad "cheater test"
echo "== bench short (solo vs cluster)"
python3 bench/pool_bench.py --rounds 2 --out bench/pool-bench-testall-$(date +%H%M).json | tail -1
[[ ${pipestatus[1]} -eq 0 ]] && ok "bench short" || bad "bench short"
echo "== bench long"
python3 bench/pool_bench.py --long 6000 --batch 64 --rounds 2 --out bench/pool-bench-testall-long-$(date +%H%M).json | tail -1
[[ ${pipestatus[1]} -eq 0 ]] && ok "bench long" || bad "bench long"
echo "== $F failed"
exit $F
