#!/bin/bash
# Quick setup script for Schwab Streamer Demo

echo "========================================"
echo "Schwab Streamer Demo - Quick Setup"
echo "========================================"
echo

# Check if API key is already set
if [ -n "$SCHWAB_API_KEY" ]; then
    echo "✓ SCHWAB_API_KEY already set"
else
    read -p "Enter your Schwab API Key: " api_key
    export SCHWAB_API_KEY="$api_key"
fi

# Check if API secret is already set
if [ -n "$SCHWAB_API_SECRET" ]; then
    echo "✓ SCHWAB_API_SECRET already set"
else
    read -sp "Enter your Schwab API Secret: " api_secret
    echo
    export SCHWAB_API_SECRET="$api_secret"
fi

# Ask about redirect URI
if [ -n "$SCHWAB_REDIRECT_URI" ]; then
    echo "✓ SCHWAB_REDIRECT_URI already set: $SCHWAB_REDIRECT_URI"
else
    echo
    echo "What is your registered redirect URI in Schwab Developer Portal?"
    echo "Examples:"
    echo "  - https://127.0.0.1:8080 (for automatic mode)"
    echo "  - https://alpha-q.com (old domain, use manual mode)"
    echo "  - https://localhost:8080"
    echo
    read -p "Redirect URI: " redirect_uri
    export SCHWAB_REDIRECT_URI="${redirect_uri:-https://127.0.0.1:8080}"
fi

# Determine mode
echo
echo "Choose authentication mode:"
echo "  1. Manual (copy-paste code from browser) - Works with ANY redirect URI"
echo "  2. Automatic (local server captures code) - Only works if redirect URI is 127.0.0.1"
echo
read -p "Choice [1/2] (default: 1): " mode_choice

if [ "$mode_choice" = "2" ]; then
    export SCHWAB_MANUAL_CODE="false"
    echo "✓ Using automatic callback mode"
else
    export SCHWAB_MANUAL_CODE="true"
    echo "✓ Using manual code entry mode"
fi

echo
echo "========================================"
echo "Configuration Summary"
echo "========================================"
echo "API Key:      ${SCHWAB_API_KEY:0:10}..."
echo "API Secret:   ${SCHWAB_API_SECRET:0:10}..."
echo "Redirect URI: $SCHWAB_REDIRECT_URI"
echo "Mode:         $([ "$SCHWAB_MANUAL_CODE" = "true" ] && echo "Manual" || echo "Automatic")"
echo

# Check dependencies
echo "========================================"
echo "Checking Dependencies"
echo "========================================"

if python3 -c "import requests" 2>/dev/null; then
    echo "✓ requests installed"
else
    echo "✗ requests not installed"
    read -p "Install now? [y/N]: " install_requests
    if [ "$install_requests" = "y" ] || [ "$install_requests" = "Y" ]; then
        pip install requests
    fi
fi

if python3 -c "import websockets" 2>/dev/null; then
    echo "✓ websockets installed"
else
    echo "✗ websockets not installed"
    read -p "Install now? [y/N]: " install_websockets
    if [ "$install_websockets" = "y" ] || [ "$install_websockets" = "Y" ]; then
        pip install websockets
    fi
fi

echo
echo "========================================"
echo "Ready to Run!"
echo "========================================"
echo
echo "Run the demo with:"
echo "  python schwap_demo.py"
echo
echo "Or to save these settings, add to your ~/.bashrc or ~/.zshrc:"
echo "  export SCHWAB_API_KEY=\"$SCHWAB_API_KEY\""
echo "  export SCHWAB_API_SECRET=\"$SCHWAB_API_SECRET\""
echo "  export SCHWAB_REDIRECT_URI=\"$SCHWAB_REDIRECT_URI\""
echo "  export SCHWAB_MANUAL_CODE=\"$SCHWAB_MANUAL_CODE\""
echo

read -p "Run the demo now? [y/N]: " run_now
if [ "$run_now" = "y" ] || [ "$run_now" = "Y" ]; then
    python schwap_demo.py
fi
