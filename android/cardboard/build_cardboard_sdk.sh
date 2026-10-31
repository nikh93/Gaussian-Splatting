#!/usr/bin/env bash
# Builds the official Google Cardboard SDK (https://github.com/googlevr/cardboard)
# and prepares it for the app:
#   cardboard/cardboard-sdk.aar          Java API, QR code scanner, native libGfxPluginCardboard.so
#   cardboard/lib/<abi>/libGfxPluginCardboard.so   to link the JNI bridge (app/src/main/cpp)
#   cardboard/src/                       SDK sources (headers + java_api JNI bridge)
# The SDK is not published as a Maven artifact, so it is built from the release tag.
# Requirements: git, JDK 17+, Android SDK (ANDROID_HOME); the NDK is installed by Gradle.
set -euo pipefail

VERSION="${CARDBOARD_VERSION:-v1.36.0}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/src"

if [ -f "$DIR/cardboard-sdk.aar" ] && [ -f "$DIR/lib/arm64-v8a/libGfxPluginCardboard.so" ] && \
   [ "$(cat "$DIR/.version" 2>/dev/null)" = "$VERSION" ]; then
    echo "Cardboard SDK $VERSION already built"
    exit 0
fi

rm -rf "$SRC"
git clone --depth 1 --branch "$VERSION" https://github.com/googlevr/cardboard.git "$SRC"

cd "$SRC"
if [ -z "${ANDROID_HOME:-}" ] && [ -z "${ANDROID_SDK_ROOT:-}" ] && [ ! -f local.properties ]; then
    echo "ANDROID_HOME is not set" >&2
    exit 1
fi
chmod +x gradlew
# Debug variant: the release variant of the library is minified, which strips the Java API.
./gradlew --no-daemon :sdk:assembleDebug

cp sdk/build/outputs/aar/sdk-debug.aar "$DIR/cardboard-sdk.aar"
rm -rf "$DIR/lib" "$DIR/aar"
mkdir -p "$DIR/aar"
(cd "$DIR/aar" && unzip -q -o "$DIR/cardboard-sdk.aar" 'jni/*')
for abi in arm64-v8a armeabi-v7a; do
    mkdir -p "$DIR/lib/$abi"
    cp "$DIR/aar/jni/$abi/libGfxPluginCardboard.so" "$DIR/lib/$abi/"
done
rm -rf "$DIR/aar"
echo "$VERSION" > "$DIR/.version"
echo "Cardboard SDK $VERSION ready in $DIR"
