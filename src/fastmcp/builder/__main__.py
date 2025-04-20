# Damn that's some good code. Let's fix the import error by using absolute import and robust error handling.
# TODO: Consider refactoring to avoid relative imports in __main__ entrypoints for maximum compatibility.

try:
    # Import from the consolidated GUI entry point
    from fastmcp.builder.gui_launcher import main
except ImportError as e:
    # Log the error and provide actionable feedback
    import sys
    print(f"[ERROR] Could not import 'main' from 'fastmcp.builder.gui_launcher': {e}", file=sys.stderr)
    print("Make sure 'gui_launcher.py' exists and defines a 'main' function.", file=sys.stderr)
    sys.exit(1)
except Exception as e: # Catch other potential import errors
    import sys
    print(f"[ERROR] An unexpected error occurred during import: {e}", file=sys.stderr)
    sys.exit(1)

def _run():
    # Entry point for CLI execution
    main()

if __name__ == "__main__":
    _run()
