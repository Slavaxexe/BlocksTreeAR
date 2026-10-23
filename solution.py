"""Platform entry: python solution.py task.md; two hours per task, time-bounded search, six workers of two threads on 8 CPUs."""

import sys
from alfa_agent.__main__ import main as cli


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    task = args.pop(0) if args and not args[0].startswith('-') else 'task.md'
    # Two hours leave a wide margin before the platform's three-hour stop; time, not a node count, ends the search.
    for option, value in {'--time-budget': '7200', '--experiment-timeout': '1200', '--max-nodes': '1000',
                          '--executors': '6', '--threads': '2'}.items():
        if not any(arg == option or arg.startswith(option + '=') for arg in args):
            args.extend((option, value))
    return cli(['baseline', '--task', task, '--platform-output', *args])


if __name__ == '__main__':
    raise SystemExit(main())
