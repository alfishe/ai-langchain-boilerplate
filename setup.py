#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from typing import List, Dict, Tuple

# Setuptools and related imports
from setuptools import setup, Command
from setuptools.command.install import install as _install

# Imports for checking prerequisites
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    print("Error: The 'packaging' library is required to run setup.py. Please install it with 'pip install packaging'")
    sys.exit(1)

# Define core dependencies in one place
INSTALL_REQUIRES = [
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "langchain-ollama>=0.0.1",
    "pyyaml>=6.0.1",
    "requests>=2.31.0",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]

def check_prerequisites() -> Tuple[Dict[str, bool], List[str]]:
    """Checks if all prerequisites are installed and working."""
    prerequisites = {"python_version": False, "ollama": False, "mistral_model": False, "dependencies": False}
    issues = []

    # 1. Check Python version
    if sys.version_info >= (3, 8):
        prerequisites["python_version"] = True
    else:
        issues.append(f"Python 3.8+ is required. You have {sys.version.split()[0]}.")

    # 2. Check if Ollama is installed and running
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=False, timeout=5
        )
        if result.returncode == 0:
            prerequisites["ollama"] = True
            # Check if mistral model is available
            if "mistral" in result.stdout.lower():
                prerequisites["mistral_model"] = True
            else:
                issues.append("Mistral model is not installed in Ollama. Please run: ollama pull mistral")
        else:
            issues.append("Ollama is not running or not installed. Please start it or install from https://ollama.ai")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        prerequisites["ollama"] = False
        issues.append("Ollama command not found. Please install from https://ollama.ai")

    # 3. Check if all required packages are installed with correct versions
    missing_packages = []
    for package_spec in INSTALL_REQUIRES:
        req = Requirement(package_spec)
        try:
            installed_ver = version(req.name)
            if not req.specifier.contains(installed_ver):
                missing_packages.append(
                    f"- {req.name}: version {installed_ver} is installed, but '{package_spec}' is required."
                )
        except PackageNotFoundError:
            missing_packages.append(f"- {req.name} is not installed (required: '{package_spec}').")

    if not missing_packages:
        prerequisites["dependencies"] = True
    else:
        issues.extend(missing_packages)
        issues.append("\nTo install required packages, run: pip install -r requirements.txt (or `python setup.py install`)")

    return prerequisites, issues

class VerifyCommand(Command):
    """Custom command to verify the environment and check prerequisites."""
    description = "Verify that all prerequisites are met"
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
        print("\nVerifying environment and prerequisites...")
        prerequisites, issues = check_prerequisites()
        all_ok = all(prerequisites.values())

        print("\n" + "="*50)
        print("Prerequisites Check Summary:")
        print(f"  [ {'✓' if prerequisites['python_version'] else '✗'} ] Python version >= 3.8")
        print(f"  [ {'✓' if prerequisites['ollama'] else '✗'} ] Ollama installed and running")
        print(f"  [ {'✓' if prerequisites['mistral_model'] else '✗'} ] Mistral model available in Ollama")
        print(f"  [ {'✓' if prerequisites['dependencies'] else '✗'} ] Required Python packages installed")
        print("="*50)

        if not all_ok:
            print("\nPrerequisite check FAILED. Please resolve the following issues:")
            for issue in issues:
                print(f"  {issue}")
            print("\nAborting.")
            sys.exit(1)
        else:
            print("\n✓ All prerequisites satisfied! Your environment is ready.")

class CustomInstallCommand(_install):
    """Custom command to run verification before the standard installation."""
    def run(self):
        print("--- Running prerequisite verification before installation ---")
        self.run_command('verify')
        print("--- Verification successful, proceeding with installation ---")
        super().run()
        print("\n✓ Package installed successfully!")

class ShowHelpCommand(Command):
    """Command to show a detailed, user-friendly help message."""
    description = "Show detailed help for this package"
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
        print("""
AI LangChain Tools Setup
======================================================================
This script helps you install and manage the 'ai-langchain-tools' package.

DEFAULT ACTION
----------------------------------------------------------------------
Running the script without any parameters will check your environment
for all necessary prerequisites.
  $ python setup.py

AVAILABLE COMMANDS
----------------------------------------------------------------------
  verify      (Default) Checks if Python version, Ollama, and all
              required packages are correctly installed.

  install     Verifies prerequisites and then installs the package
              in editable mode. This is the recommended way to install.

  help        Shows this help message.

QUICK START
----------------------------------------------------------------------
1. Verify your environment (this is the default action):
   $ python setup.py

2. Install the package:
   $ python setup.py install
""")

# If no command is provided, we are forcing 'verify' to be the default.
if len(sys.argv) == 1:
    sys.argv.append('verify')

def main():
    """Entry point for the verify-env script."""
    prerequisites, issues = check_prerequisites()
    all_ok = all(prerequisites.values())
    
    if not all_ok:
        print("\nPrerequisite check FAILED. Please resolve the following issues:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("\n✓ All prerequisites satisfied! Your environment is ready.")
        sys.exit(0)

if __name__ == "__main__":
    main()