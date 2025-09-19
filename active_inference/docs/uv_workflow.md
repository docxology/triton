# uv Workflow Guide

This guide explains how to use uv for package management and development workflow with the Active Inference Framework.

## What is uv?

uv is a fast Python package installer and resolver, written in Rust. It provides:

- **Lightning-fast installation**: 10-100x faster than pip
- **Reliable dependency resolution**: Never guess what version will be installed
- **Virtual environment management**: Automatic venv creation and management
- **Reproducible builds**: Lock files ensure consistent environments
- **Script execution**: Run commands in the project's virtual environment

## Installation

### Install uv

```bash
# Using pip (temporary)
pip install uv

# Or using Homebrew (macOS/Linux)
brew install uv

# Or using cargo (if you have Rust installed)
cargo install uv
```

### Verify Installation

```bash
uv --version
```

## Project Setup

### Initial Setup

```bash
# Navigate to the project directory
cd active_inference

# Sync core dependencies (creates .venv automatically)
uv sync

# Sync with development dependencies
uv sync --dev

# Install Triton for GPU acceleration
uv add triton

# Or install everything at once
uv sync --dev && uv add triton
```

### Automated Setup

Use the provided setup script:

```bash
# Run the automated setup
./setup_uv.sh
```

This script will:
- Check uv installation
- Sync dependencies
- Optionally install Triton
- Run tests to verify installation
- Create an activation script

## Development Workflow

### Running Commands

All commands should be prefixed with `uv run` to execute them in the project's virtual environment:

```bash
# Run tests
uv run pytest tests/

# Run examples
uv run python run_all_examples.py

# Run validation
uv run python validate_triton_usage.py

# Run linters
uv run black .
uv run isort .
uv run flake8 .

# Type checking
uv run mypy src/
```

### Defined Scripts

The project includes predefined scripts in `pyproject.toml`:

```bash
# Testing
uv run test              # Run tests
uv run test-verbose      # Run tests with verbose output
uv run test-coverage     # Run tests with coverage

# Development
uv run examples          # Run all examples
uv run validate          # Run Triton validation
uv run lint              # Run all linters
uv run format            # Format code
uv run type-check        # Run type checking

# Package management
uv run install           # Sync dependencies
uv run install-dev       # Sync with dev dependencies
uv run install-triton    # Install Triton
uv run install-all       # Install everything

# Build and publish
uv run build             # Build package
uv run publish           # Publish package
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add with specific version
uv add package-name==1.2.3

# Remove dependency
uv remove package-name
```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific package
uv sync --upgrade-package package-name

# Update to latest compatible version
uv sync --upgrade-package package-name --upgrade
```

## Environment Management

### Virtual Environment

uv automatically creates and manages a virtual environment in `.venv/`:

```bash
# The virtual environment is created automatically
# Activate it manually if needed
source .venv/bin/activate

# Deactivate
deactivate
```

### Python Version

The project specifies Python 3.8+ in `pyproject.toml`. uv will use the appropriate Python version:

```bash
# Check which Python version uv is using
uv run python --version

# Use a specific Python version
uv sync --python 3.9
```

## Lock File Management

### Understanding uv.lock

The `uv.lock` file ensures reproducible environments:

```bash
# The lock file is created automatically during sync
# It contains exact versions of all dependencies

# To update the lock file
uv sync --upgrade

# To recreate from scratch
rm uv.lock
uv sync
```

### Resolving Conflicts

If you encounter dependency conflicts:

```bash
# Try upgrading conflicting packages
uv sync --upgrade-package package-name

# Or remove and re-add problematic packages
uv remove package-name
uv add package-name

# Check dependency tree
uv tree
```

## Performance Tips

### Fast Installation

uv is designed for speed:

```bash
# Parallel downloads
uv sync --parallel

# Use cached packages
uv sync --cache-dir ~/.cache/uv
```

### Workspace Management

For monorepo setups:

```bash
# Add workspace members
uv add --workspace package-name

# Sync entire workspace
uv sync
```

## Troubleshooting

### Common Issues

1. **uv command not found**
   ```bash
   pip install uv
   ```

2. **Permission errors**
   ```bash
   # Use --user flag or virtual environment
   uv sync --user
   ```

3. **Dependency conflicts**
   ```bash
   # Clear cache and retry
   uv cache clean
   uv sync
   ```

4. **Python version issues**
   ```bash
   # Specify Python version explicitly
   uv sync --python python3.9
   ```

### Getting Help

```bash
# Show uv help
uv --help

# Show command-specific help
uv sync --help

# Check uv version and installation
uv version
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
        with:
          version: "latest"
      - name: Install dependencies
        run: uv sync --dev
      - name: Run tests
        run: uv run pytest tests/
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: uv-format
        name: Format with uv
        entry: uv run black .
        language: system
        pass_filenames: false
      - id: uv-type-check
        name: Type check with uv
        entry: uv run mypy src/
        language: system
        pass_filenames: false
```

## Best Practices

1. **Always use `uv run`** for commands to ensure proper environment isolation
2. **Commit `uv.lock`** to ensure reproducible environments across team members
3. **Use version constraints** in `pyproject.toml` for stability
4. **Regularly update dependencies** with `uv sync --upgrade`
5. **Use workspace features** for monorepo setups

## Integration with IDEs

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. Open project
2. Go to File > Settings > Project > Python Interpreter
3. Add interpreter from `.venv/bin/python`

## Migration from pip

### Converting from requirements.txt

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Convert to pyproject.toml format
# Edit pyproject.toml to include dependencies
uv sync
```

### Converting from virtualenv

```bash
# Old workflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# New workflow
uv sync --dev
uv run python your_script.py
```

This guide provides a comprehensive overview of using uv with the Active Inference Framework. For more advanced features, refer to the [official uv documentation](https://docs.astral.sh/uv/).
