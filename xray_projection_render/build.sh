#!/bin/bash

# Exit on error
set -e

# Print commands
set -x

# Create a build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

# Function to build for a specific platform
build_for_platform() {
    local GOOS=$1
    local GOARCH=$2
    local OUTPUT_NAME=$3
    
    echo "Building for $GOOS/$GOARCH..."
    GOOS=$GOOS GOARCH=$GOARCH go build -o "$BUILD_DIR/$OUTPUT_NAME" main.go api.go
    echo "Build complete for $GOOS/$GOARCH"
}

# Function to build shared library for a specific platform
build_shared_library() {
    local GOOS=$1
    local GOARCH=$2
    
    echo "Building shared library for $GOOS/$GOARCH..."
    
    # go build with -buildmode=c-shared outputs both .so/.dylib/.dll and .h files
    # The -o flag sets the base name for both outputs
    # Go automatically adds the correct extension (.so, .dylib, or .dll) and creates a .h file
    OUTPUT_BASE="$BUILD_DIR/libxray_projection_render"
    
    GOOS=$GOOS GOARCH=$GOARCH go build -buildmode=c-shared -o "$OUTPUT_BASE" .
    
    # On Linux, ensure the .so extension is present (Go may create it without extension)
    if [ "$GOOS" = "linux" ]; then
        if [ -f "$OUTPUT_BASE" ] && [ ! -f "$OUTPUT_BASE.so" ]; then
            mv "$OUTPUT_BASE" "$OUTPUT_BASE.so"
            echo "Renamed library file to include .so extension"
        fi
    fi
    
    echo "Shared library build complete for $GOOS/$GOARCH"
    echo "  Library: ${OUTPUT_BASE}.* (extension depends on platform)"
    echo "  Header: ${OUTPUT_BASE}.h"
}

# Function to get output name for a platform
get_output_name() {
    local GOOS=$1
    local GOARCH=$2
    
    case "$GOOS" in
        darwin)
            echo "xray_projection_render_darwin-${GOARCH}"
            ;;
        windows)
            echo "xray_projection_render_windows-${GOARCH}.exe"
            ;;
        linux)
            echo "xray_projection_render_linux-${GOARCH}"
            ;;
        *)
            echo "xray_projection_render_${GOOS}-${GOARCH}"
            ;;
    esac
}

# Detect current platform
CURRENT_OS=$(go env GOOS)
CURRENT_ARCH=$(go env GOARCH)

# Check if --all flag is set
BUILD_ALL=false
if [[ "$1" == "--all" ]] || [[ "$1" == "-a" ]]; then
    BUILD_ALL=true
fi

if [ "$BUILD_ALL" = true ]; then
    echo "Building for all platforms..."
    
    # Build for Apple Silicon (darwin/arm64)
    build_for_platform "darwin" "arm64" "xray_projection_render_darwin-arm64"
    
    # Build for Windows (windows/amd64)
    build_for_platform "windows" "amd64" "xray_projection_render_windows-amd64.exe"
    
    # Build for Linux (linux/amd64)
    build_for_platform "linux" "amd64" "xray_projection_render_linux-amd64"
    
    echo "All platform builds completed!"
else
    echo "Building for current platform: $CURRENT_OS/$CURRENT_ARCH"
    OUTPUT_NAME=$(get_output_name "$CURRENT_OS" "$CURRENT_ARCH")
    build_for_platform "$CURRENT_OS" "$CURRENT_ARCH" "$OUTPUT_NAME"
fi

# Build shared library for current platform
echo "Building shared library for current platform..."
build_shared_library "$CURRENT_OS" "$CURRENT_ARCH"

echo "Build completed successfully!"
echo "Build artifacts are in the $BUILD_DIR directory"
echo "Shared library: $BUILD_DIR/libxray_projection_render" 