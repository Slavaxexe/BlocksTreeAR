"""Platform entry: python solution.py task.md [--max-nodes 4] [--key-stdin]."""

import sys
from alfa_agent.__main__ import main as cli


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    task = args.pop(0) if args and not args[0].startswith('-') else 'task.md'
    return cli(['baseline', '--task', task, *args])


if __name__ == '__main__':
    raise SystemExit(main())
