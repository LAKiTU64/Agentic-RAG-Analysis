#!/usr/bin/env python3
"""AI Agent Quick Startup Script"""

import sys
import subprocess
from pathlib import Path

def main():
    print("AI Agent - Startup Wizard")
    print("="*50)
    print()
    print("Choose version:")
    print("1. LangChain Version (Recommended)")
    print("2. Original Version")
    print()
    
    choice = input("Enter option (1/2) [Default: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\nStarting LangChain Version...")
        backend_path = Path("AI agent/langchain_version/web_langchain_backend.py")
    elif choice == "2":
        print("\nStarting Original Version...")
        backend_path = Path("AI agent/original_version/web_agent_backend.py")
    else:
        print("Invalid option")
        return
    
    if not backend_path.exists():
        print(f"Error: File not found: {backend_path}")
        return
    
    print(f"Service URL: http://localhost:8000")
    print(f"Chat: http://localhost:8000/chat")
    print("Press Ctrl+C to stop")
    print("="*50)
    
    try:
        subprocess.run([sys.executable, str(backend_path)], check=True)
    except KeyboardInterrupt:
        print("\nService stopped")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
