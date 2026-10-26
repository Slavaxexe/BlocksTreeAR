"""Platform entry: python solution.py path/to/task.md [--max-experiments 10]."""

import sys
from agent.__main__ import main as cli


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    task = args.pop(0) if args and not args[0].startswith('-') else 'task.md'
    return cli(['search', '--task', task, *args])


if __name__ == '__main__':
    raise SystemExit(main())
