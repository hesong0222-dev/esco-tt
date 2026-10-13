#!/bin/bash
# waits for an orphaned TT eval (pid 21362, sisco cha-root eval started by the old queue) to finish, then runs queue.sh
cd "$(dirname "$0")"
while kill -0 21362 2>/dev/null; do sleep 30; done
exec ./queue.sh
