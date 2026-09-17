#!/bin/bash
# The exact noise schedules used by every calibrated arm are committed as one
# tarball (13 MB) instead of ~120 MB of JSON. Run once after cloning:
#   bash analysis/unpack_schedules.sh
cd "$(dirname "$0")" && tar xzf noise_schedules.tar.gz && ls noise_schedule_*.json | wc -l | xargs echo "unpacked schedules:"
