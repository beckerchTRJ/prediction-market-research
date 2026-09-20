#!/usr/bin/env bash
# Install cron entries for automated collection + settlement.
# Collect 8x/day (every 3h) to capture price evolution incl. intraday nowcast;
# settle twice daily. Adjust cadence to taste.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v 'kalshi_weather_trading/scripts' > "$TMP" || true
{
  echo "0 */3 * * * $ROOT/scripts/run_collect.sh"
  echo "30 13,17 * * * $ROOT/scripts/run_settle.sh"
} >> "$TMP"
crontab "$TMP"
rm -f "$TMP"
echo "Installed cron entries:"
crontab -l | grep kalshi_weather_trading
