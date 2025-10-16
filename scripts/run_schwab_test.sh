#!/bin/bash
# Quick start script for testing Schwab smart caching
#
# This script helps you set up and run the Schwab AAPL test

set -e

echo "=========================================="
echo "  Schwab Smart Caching Test - Quick Start"
echo "=========================================="
echo ""

# Check if env vars are set
if [ -z "$SCHWAB_API_KEY" ]; then
    echo "⚠️  SCHWAB_API_KEY not set"
    echo ""
    echo "Please provide your Schwab API credentials:"
    read -p "Schwab App Key (client_id): " api_key
    read -p "Schwab Secret (client_secret): " api_secret
    echo ""

    export SCHWAB_API_KEY="$api_key"
    export SCHWAB_API_SECRET="$api_secret"
    export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"

    echo "✓ Environment variables set (for this session)"
    echo ""
    echo "To make permanent, add to ~/.zshrc:"
    echo "  export SCHWAB_API_KEY=\"$api_key\""
    echo "  export SCHWAB_API_SECRET=\"$api_secret\""
    echo ""
else
    echo "✓ SCHWAB_API_KEY is set"
    echo "✓ SCHWAB_API_SECRET is set"
    echo ""
fi

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Running Schwab AAPL test..."
echo ""
echo "Note: Browser will open for authentication (first time only)"
echo "      You may see SSL certificate warning (this is expected)"
echo "      Click 'Advanced' → 'Proceed' to continue"
echo ""

# Run the test
python scripts/test_schwab_aapl.py

echo ""
echo "=========================================="
echo "  Test Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check cache: qtrader data cache-info --dataset schwab-us-equity-1d-adjusted"
echo "  2. Update data: qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL"
echo "  3. Browse data: qtrader data raw --symbol AAPL --start-date 2024-01-01 --end-date 2024-12-31 --source schwab"
echo ""
