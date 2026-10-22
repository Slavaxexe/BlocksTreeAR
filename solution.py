"""Platform entry: python solution.py task.md; 175 minutes, 100 nodes, two workers."""

import sys
from alfa_agent.__main__ import main as cli


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    task = args.pop(0) if args and not args[0].startswith('-') else 'task.md'
    # Leave five minutes before the platform's hard stop; explicit CLI settings take precedence.
    for option, value in {'--time-budget': '10500', '--experiment-timeout': '1200', '--max-nodes': '100'}.items():
        if not any(arg == option or arg.startswith(option + '=') for arg in args):
            args.extend((option, value))
    return cli(['baseline', '--task', task, '--platform-output', *args])


if __name__ == '__main__':
    raise SystemExit(main())
