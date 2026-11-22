#!/bin/zsh
# Build "ANE Embed.app" (menu bar, no Dock icon) from the SwiftPM release build.
# APP_NAME / APP_ID / APP_VERSION build a separate bundle beside it, e.g. a new version next to the
# running one: APP_NAME="ANE Embed 0.2" APP_ID=studio.soulbrews.ane-embed.v02 zsh make_app.sh
# (its own settings domain, so it can serve on another port while the old one keeps running).
# The pool app is "Chippy": APP_NAME=Chippy APP_ID=studio.soulbrews.ane-embed.v03 zsh make_app.sh
# Version is CalVer, yy.m.d of the build day (Asia/Bangkok), build number H*100+M (09:37 -> 937);
# APP_VERSION overrides it.
# Assets are not bundled: the app reads ~/Library/Application Support/ANEEmbed/embeddinggemma-w16,
# written by export_assets.py (copy that folder to another Mac, with --copy-packages).
set -euo pipefail
cd "${0:A:h}"
(cd tokenizer-ffi && cargo build --release)
swift build -c release
NAME="${APP_NAME:-ANE Embed}"
ID="${APP_ID:-studio.soulbrews.ane-embed}"
VERSION="${APP_VERSION:-$(TZ=Asia/Bangkok date +%y.%-m.%-d)}"
BUILD="$(TZ=Asia/Bangkok date +%-H%M)"
APP="$NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp .build/release/ANEEmbedBar "$APP/Contents/MacOS/ANEEmbedBar"
cp .build/release/ane-embed "$APP/Contents/MacOS/ane-embed"   # handed to Macs that join the pool
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$NAME</string>
  <key>CFBundleIdentifier</key><string>$ID</string>
  <key>CFBundleExecutable</key><string>ANEEmbedBar</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <!-- workers are plain http on loopback/Tailscale, and only URLs the user adds -->
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsArbitraryLoads</key><true/></dict>
  <key>NSLocalNetworkUsageDescription</key><string>Serves embeddings to other Macs on your Tailscale network.</string>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
echo "built $PWD/$APP"
