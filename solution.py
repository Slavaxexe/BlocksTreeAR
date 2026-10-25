"""Platform entry: python solution.py task.md; 160 minutes per task, time-bounded search, six workers of one thread on 8 CPUs."""

from pathlib import Path
import sys

from alfa_agent.__main__ import main as cli


# Explicit opt-in: default submissions must come from the main agent only.
ENABLE_FALLBACK = False


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    task = args.pop(0) if args and not args[0].startswith('-') else 'task.md'
    if not ENABLE_FALLBACK:
        return run(task, args)
    try:  # an answer without the LLM first (about a minute); the agent overwrites it with any better model
        from alfa_agent.fallback import emergency
        emergency(Path(task))
    except Exception as error:
        print(f'Initial fallback submission failed: {type(error).__name__}: {error}', file=sys.stderr)
    destination = Path(task).resolve().parent / 'submission.csv'
    try:
        code = run(task, args)
    finally:  # whatever happened above, the platform scores a submission.csv next to task.md
        if not destination.exists():
            try:
                from alfa_agent.fallback import emergency
                emergency(Path(task))
            except Exception as error:
                print(f'Fallback submission failed: {type(error).__name__}: {error}', file=sys.stderr)
    return 0 if destination.exists() else code


def run(task, args):
    # Leave twenty minutes before the platform's three-hour stop; time, not a node count, ends the search.
    for option, value in {'--time-budget': '9600', '--experiment-timeout': '900', '--max-nodes': '1000',
                          '--executors': '6', '--threads': '1', '--llm-concurrency': '8', '--research-parallel': '2'}.items():
        if not any(arg == option or arg.startswith(option + '=') for arg in args):
            args.extend((option, value))
    return cli(['baseline', '--task', task, '--platform-output', *args])


if __name__ == '__main__':
    raise SystemExit(main())
