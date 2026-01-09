#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent File Path Fixer

Fix:
1. Static file paths
2. Python import paths
3. Directory structure
4. File locations
"""

import os
import shutil
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def create_directories():
    print_section("Creating Directory Structure")
    
    directories = [
        "workspace/models",
        "analysis_results",
        "SGlang",
        "AI agent/langchain_version/static",
        "AI agent/original_version/static",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    sglang_readme = Path("SGlang/README.md")
    if not sglang_readme.exists():
        sglang_readme.write_text("# SGlang Directory\n\nPlace your SGlang code here.\n", encoding='utf-8')
        print(f"Created: {sglang_readme}")

def copy_static_files():
    print_section("Copying Static Files")
    
    source = Path("AI agent/web_interface/static/chat.html")
    
    targets = [
        Path("AI agent/langchain_version/static/chat.html"),
        Path("AI agent/original_version/static/chat.html"),
    ]
    
    if source.exists():
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"Copied: {source} -> {target}")
    else:
        print(f"Warning: Source file not found: {source}")

def fix_import_paths():
    print_section("Fixing Import Paths")
    
    langchain_agent_file = Path("AI agent/langchain_version/langchain_agent.py")
    
    if langchain_agent_file.exists():
        content = langchain_agent_file.read_text(encoding='utf-8')
        
        if "sys.path.insert" not in content:
            lines = content.split('\n')
            
            import_line_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_line_idx = i
                    break
            
            path_setup = """# Path setup for imports
import sys
from pathlib import Path
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir.parent / 'original_version'))
sys.path.insert(0, str(_current_dir.parent.parent / 'TOOLS' / 'Auto_Anlyze_tool'))
"""
            lines.insert(import_line_idx, path_setup)
            
            new_content = '\n'.join(lines)
            langchain_agent_file.write_text(new_content, encoding='utf-8')
            print(f"Fixed: {langchain_agent_file}")
        else:
            print(f"Already fixed: {langchain_agent_file}")
    
    ai_agent_file = Path("AI agent/original_version/ai_agent_analyzer.py")
    
    if ai_agent_file.exists():
        content = ai_agent_file.read_text(encoding='utf-8')
        
        if 'tools_dir = Path("TOOLS/Auto_Anlyze_tool")' in content:
            content = content.replace(
                'tools_dir = Path("TOOLS/Auto_Anlyze_tool")',
                'tools_dir = Path(__file__).parent.parent.parent / "TOOLS" / "Auto_Anlyze_tool"'
            )
            
            ai_agent_file.write_text(content, encoding='utf-8')
            print(f"Fixed: {ai_agent_file}")
        else:
            print(f"Already fixed: {ai_agent_file}")

def create_startup_script():
    print_section("Creating Startup Script")
    
    start_script = Path("start_ai_agent.py")
    script_content = '''#!/usr/bin/env python3
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
        print("\\nStarting LangChain Version...")
        backend_path = Path("AI agent/langchain_version/web_langchain_backend.py")
    elif choice == "2":
        print("\\nStarting Original Version...")
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
        print("\\nService stopped")
    except Exception as e:
        print(f"\\nError: {e}")

if __name__ == "__main__":
    main()
'''
    start_script.write_text(script_content, encoding='utf-8')
    print(f"Created: {start_script}")

def create_requirements():
    print_section("Creating Requirements File")
    
    requirements = Path("requirements_complete.txt")
    req_content = '''# AI Agent Dependencies

# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0

# Data Processing
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0
requests>=2.31.0

# LangChain (Optional)
langchain>=0.0.350
langchain-openai>=0.0.2
langchain-community>=0.0.10

# Other
python-multipart>=0.0.6
'''
    requirements.write_text(req_content, encoding='utf-8')
    print(f"Created: {requirements}")

def verify_environment():
    print_section("Verifying Environment")
    
    critical_files = [
        "AI agent/configs_and_docs/agent_config.yaml",
        "AI agent/langchain_version/langchain_agent.py",
        "AI agent/langchain_version/web_langchain_backend.py",
        "AI agent/original_version/ai_agent_analyzer.py",
        "TOOLS/Auto_Anlyze_tool/nsys_parser.py",
        "TOOLS/Auto_Anlyze_tool/ncu_parser.py",
    ]
    
    all_exist = True
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"OK: {file_path}")
        else:
            print(f"MISSING: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    print("""
===============================================================
   AI Agent - Path Fixer
===============================================================
""")
    
    try:
        create_directories()
        copy_static_files()
        fix_import_paths()
        create_startup_script()
        create_requirements()
        all_ok = verify_environment()
        
        print_section("Fix Complete")
        
        if all_ok:
            print("All critical files verified")
            print()
            print("Next Steps:")
            print("1. pip install -r requirements_complete.txt")
            print("2. Configure SGlang (place in SGlang/ directory)")
            print("3. Configure models (place in workspace/models/)")
            print("4. python start_ai_agent.py")
            print("5. Open http://localhost:8000/chat")
            print()
            print("See: 配置指南.md for details")
        else:
            print("Some files are missing")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

