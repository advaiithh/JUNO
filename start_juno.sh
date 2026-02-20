#!/bin/bash

echo "========================================="
echo "   JUNO AI Voice Assistant Setup"
echo "========================================="
echo ""

# Check if already registered
if [ -f "registered_faces_advanced.pkl" ]; then
    echo "✓ Face already registered"
    echo ""
    echo "Starting JUNO server..."
    python server.py
else
    echo "⚠️  No registered face found!"
    echo ""
    echo "Please register your face first:"
    echo "  python recognition_advanced.py"
    echo ""
    echo "After registration, run this script again."
fi
