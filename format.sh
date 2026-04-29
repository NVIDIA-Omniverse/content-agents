#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# format.sh - Code formatting and linting script for World Understanding
#
# This script runs various code quality tools to ensure the codebase
# conforms to the project's style guidelines defined in pyproject.toml.
#
# Usage:
#   ./format.sh          # Run ruff linting/formatting and fix issues
#   ./format.sh check    # Only check, don't fix (for CI)
#   ./format.sh fix      # Fix all issues (default)
#   ./format.sh --mypy   # Also run mypy type checking (optional)
#
# Options can be combined:
#   ./format.sh check --mypy
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
MODE="fix"  # Default to fix mode
RUN_MYPY=false  # mypy is optional, off by default

# Parse arguments
for arg in "$@"; do
    case $arg in
        check)
            MODE="check"
            ;;
        fix)
            MODE="fix"
            ;;
        --mypy)
            RUN_MYPY=true
            ;;
    esac
done

# Print functions
print_header() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running in virtual environment
check_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        print_warning "Not running in a virtual environment!"
        print_info "Consider activating your virtual environment first:"
        print_info "  source .venv/bin/activate"
        echo
    fi
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Install formatting tools if needed
ensure_tools() {
    local missing_tools=()

    if ! command_exists ruff; then
        missing_tools+=("ruff")
    fi

    if [ "$RUN_MYPY" = true ] && ! command_exists mypy; then
        missing_tools+=("mypy")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_warning "Missing tools: ${missing_tools[*]}"
        print_info "Installing missing tools..."
        pip install -q "${missing_tools[@]}"
        echo
    fi
}

# Run ruff linter
run_ruff_lint() {
    print_header "Running Ruff Linter"
    
    if [ "$MODE" = "check" ]; then
        print_info "Checking for linting issues..."
        if ruff check "$PROJECT_ROOT"; then
            print_success "No linting issues found"
        else
            print_error "Linting issues found"
            return 1
        fi
    else
        print_info "Fixing linting issues..."
        if ruff check --fix --unsafe-fixes "$PROJECT_ROOT"; then
            print_success "Linting complete"
        else
            print_warning "Some issues could not be auto-fixed"
            print_info "Run 'ruff check $PROJECT_ROOT' to see remaining issues"
        fi
    fi
}

# Run ruff formatter
run_ruff_format() {
    print_header "Running Ruff Formatter"
    
    if [ "$MODE" = "check" ]; then
        print_info "Checking code formatting..."
        if ruff format --check "$PROJECT_ROOT"; then
            print_success "Code is properly formatted"
        else
            print_error "Formatting issues found"
            return 1
        fi
    else
        print_info "Formatting code..."
        ruff format "$PROJECT_ROOT"
        print_success "Code formatted"
    fi
}

# Run mypy type checker
run_mypy() {
    print_header "Running MyPy Type Checker"
    
    print_info "Checking type annotations..."
    
    # Run mypy on the main package
    if mypy world_understanding; then
        print_success "Type checking passed"
    else
        print_error "Type checking failed"
        if [ "$MODE" = "check" ]; then
            return 1
        else
            print_info "Type errors must be fixed manually"
        fi
    fi
}

# Check for common issues
run_custom_checks() {
    print_header "Running Custom Checks"
    
    # Check for trailing whitespace
    print_info "Checking for trailing whitespace..."
    if grep -r '[[:space:]]$' --include="*.py" "$PROJECT_ROOT/world_understanding" "$PROJECT_ROOT/tests" 2>/dev/null; then
        print_warning "Found trailing whitespace in files above"
        if [ "$MODE" = "fix" ]; then
            print_info "Removing trailing whitespace..."
            find "$PROJECT_ROOT/world_understanding" "$PROJECT_ROOT/tests" -name "*.py" -type f -exec sed -i 's/[[:space:]]*$//' {} +
            print_success "Trailing whitespace removed"
        fi
    else
        print_success "No trailing whitespace found"
    fi
    
    # Check for print statements in non-example code
    print_info "Checking for print statements..."
    if grep -r '^[[:space:]]*print(' --include="*.py" "$PROJECT_ROOT/world_understanding" 2>/dev/null | grep -v '# noqa'; then
        print_warning "Found print statements in code (consider using logging instead)"
    else
        print_success "No print statements found"
    fi
}

# Summary of checks
print_summary() {
    print_header "Formatting Rules"
    echo "The formatting rules are defined in pyproject.toml:"
    echo ""
    echo "  [tool.ruff]"
    echo "  - Line length: 88 characters"
    echo "  - Python version: 3.10+"
    echo "  - Enabled rules: E, W, F, I, B, C4, UP"
    echo "  - Ignored: E501 (line too long), B008, C901"
    echo ""
    echo "  [tool.mypy]"
    echo "  - Python version: 3.12"
    echo "  - Strict type checking enabled"
    echo "  - No implicit Optional"
    echo ""
    print_info "See pyproject.toml for complete configuration"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    echo -e "${BOLD}${BLUE}World Understanding Code Formatter${NC}"
    echo -e "${BLUE}Mode: $MODE${NC}\n"
    
    # Check environment
    check_venv
    
    # Ensure tools are installed
    ensure_tools
    
    # Track if any checks failed
    local failed=0
    
    # Run all checks
    run_ruff_lint || failed=1
    run_ruff_format || failed=1
    if [ "$RUN_MYPY" = true ]; then
        run_mypy || failed=1
    fi
    run_custom_checks || failed=1
    
    # Print summary
    echo
    if [ $failed -eq 0 ]; then
        print_success "All checks passed! 🎉"
    else
        print_error "Some checks failed"
        if [ "$MODE" = "check" ]; then
            print_info "Run './format.sh fix' to automatically fix issues"
        fi
    fi
    
    # Show where rules are defined
    echo
    print_summary
    
    # Exit with appropriate code
    exit $failed
}

# Run main function
main
