"""Management script dispatcher."""

from argparse import ArgumentParser, RawTextHelpFormatter
from tools import commands


if __name__ == '__main__':
    parser = ArgumentParser(
        description="DART Preprocessing, Evaluation & Visualization scripts.",
        formatter_class=RawTextHelpFormatter)

    subparsers = parser.add_subparsers()
    for name, command in commands.items():
        p = subparsers.add_parser(
            name, help=command.__doc__, description=command.__doc__,
            formatter_class=RawTextHelpFormatter)
        command._parse(p)
        p.set_defaults(_func=command._main)

    args = parser.parse_args()
    args._func(args)
