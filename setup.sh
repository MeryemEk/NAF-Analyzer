#!/bin/bash

# French Sector Analyzer - Setup Script
# This script sets up the environment for running the Streamlit app

echo "=========================================="
echo "French Sector Analyzer - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Install requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Dependencies installed successfully!"
echo ""

# Create sample data if needed
if [ ! -f "popdept.xlsx" ]; then
    echo "Creating sample population data..."
    python3 create_sample_data.py
    echo ""
fi

# Create .streamlit directory if needed
if [ ! -d ".streamlit" ]; then
    mkdir .streamlit
    echo "Created .streamlit directory"
fi

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the app, run:"
echo "  streamlit run streamlit_app.py"
echo ""
echo "Or use the quick start command:"
echo "  make run"
echo ""
echo "For detailed instructions, see:"
echo "  - README.md (comprehensive documentation)"
echo "  - QUICKSTART.md (step-by-step guide)"
echo ""
