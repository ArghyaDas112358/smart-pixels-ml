#!/bin/bash
# Wrapper: run the watcher with the AF MOTD banner filtered out of the event
# stream (every AF shell prints it; without the filter each monitor cycle
# would emit banner noise as "events").
SCDIR=/tmp/claude-978920/-work-users-das214-SmartPixels/7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad
bash "$SCDIR/watch_runs.sh" 2>&1 | grep --line-buffered -vE "║|╔|╚|╠|═|Mattermost|Pixi|pixi shell|^\s*$|To activate|To deactivate|exit\s*$"
