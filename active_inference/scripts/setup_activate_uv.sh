#!/bin/bash

# Active Inference Framework - Unified Setup and Activation Script
# This script handles both environment setup and activation for the uv-based development environment

set -e

# Function to check if setup is already complete
is_setup_complete() {
    [ -f ".venv/bin/python" ] || [ -f ".venv/Scripts/python.exe" ] || [ -d ".venv" ]
}

# Function to perform initial setup
perform_setup() {
    echo "🤖 Active Inference Framework - Initial Setup"
    echo "==========================================="

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "❌ uv is not installed. Please install uv first:"
        echo "   pip install uv"
        echo "   or"
        echo "   brew install uv"
        echo "   or"
        echo "   cargo install uv"
        exit 1
    fi

    echo "✅ uv is installed"

    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        echo "❌ pyproject.toml not found. Please run this script from the active_inference directory"
        exit 1
    fi

    echo "✅ Found pyproject.toml"

    # Sync dependencies
    echo ""
    echo "📦 Syncing core dependencies..."
    uv sync

    echo ""
    echo "📦 Syncing development dependencies..."
    uv sync --dev

    # Check platform for Triton compatibility
    if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
        echo "⚠️  Apple Silicon Mac detected - Building Triton from source for MPS support"
        echo "📦 Installing build dependencies..."
        uv add --dev cmake lit ninja

        echo "🔨 Building Triton from local source..."
        # Set PYTHONPATH to include the local Triton installation
        export PYTHONPATH="$(pwd)/../python:$PYTHONPATH"
        cd ..
        if pip install -e python/; then
            echo "✅ Triton built and installed successfully"
            export TRITON_AVAILABLE=true
        else
            echo "⚠️  Triton build failed - using PyTorch fallbacks"
            export TRITON_AVAILABLE=false
        fi
        cd active_inference
    else
        # For CUDA platforms, install from source as well
        echo "🤔 Installing Triton from source for GPU acceleration..."
        echo "📦 Installing build dependencies..."
        uv add --dev cmake lit ninja

            echo "🔨 Building Triton from local source..."
            export PYTHONPATH="$(pwd)/../python:$PYTHONPATH"
        cd ..
        if pip install -e python/; then
            echo "✅ Triton built and installed successfully"
            export TRITON_AVAILABLE=true
        else
            echo "⚠️  Triton build failed - using PyTorch fallbacks"
            export TRITON_AVAILABLE=false
        fi
        cd active_inference
    fi

    # Run tests to verify installation
    echo ""
    echo "🧪 Running tests to verify installation..."
    if uv run pytest tests/ --tb=short -q; then
        echo "✅ Tests passed!"
    else
        echo "⚠️  Some tests failed. This might be expected if Triton is not installed."
    fi

    echo ""
    echo "🎉 Initial setup complete!"
    echo ""
}

# Function to activate environment
activate_environment() {
    if command -v uv &> /dev/null; then
        export UV_ACTIVE=1

                # Set PYTHONPATH to include local Triton installation
                export PYTHONPATH="$(pwd)/../python:$PYTHONPATH"

        # Check if Triton is available
        if python -c "import triton; print('Triton version:', triton.__version__)" 2>/dev/null; then
            echo "✅ uv environment activated with Triton support"
            echo "   Triton version: $(python -c "import triton; print(triton.__version__)")"
            export TRITON_AVAILABLE=true
        else
            echo "✅ uv environment activated (Triton not available)"
            echo "   Using PyTorch fallbacks for GPU acceleration"
            export TRITON_AVAILABLE=false
        fi

        echo "   Use 'uv run' to execute commands in the environment"
        echo "   Example: uv run python run_all_examples.py"
        echo ""
        echo "📚 Available uv scripts (defined in pyproject.toml):"
        echo "   uv run test          - Run tests"
        echo "   uv run examples      - Run examples"
        echo "   uv run validate      - Run validation"
        echo "   uv run lint          - Run linters"
        echo "   uv run format        - Format code"
        echo "   uv run type-check    - Run type checking"
        echo ""
        echo "📋 Quick commands:"
        echo "   uv run python run_all_tests.py"
        echo "   uv run python run_all_examples.py"
        echo "   uv run python src/validate_triton_usage.py"
        echo "   uv run python src/triton_analysis_demo.py"
    else
        echo "❌ uv not found"
        exit 1
    fi
}

# Main script logic
case "${1:-}" in
    "setup")
        echo "🔧 Running setup only..."
        perform_setup
        ;;
    "activate")
        if ! is_setup_complete; then
            echo "⚠️  Setup not complete. Running setup first..."
            perform_setup
        fi
        echo "🔄 Activating environment..."
        activate_environment
        ;;
    "force-setup")
        echo "🔧 Forcing complete setup..."
        perform_setup
        echo "🔄 Activating environment..."
        activate_environment
        ;;
    *)
        # Default behavior: setup if needed, then activate
        if ! is_setup_complete; then
            echo "🔧 Setup not complete. Running initial setup..."
            perform_setup
        else
            echo "✅ Setup already complete"
        fi
        echo "🔄 Activating environment..."
        activate_environment
        ;;
esac

echo ""
echo "📁 Environment status:"
echo "   - .venv/            - Virtual environment"
echo "   - uv.lock           - Lock file"
echo "   - src/              - Source code and scripts"
