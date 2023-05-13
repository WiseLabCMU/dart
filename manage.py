"""Management script dispatcher."""

from argparse import ArgumentParser
from tools import commands


if __name__ == '__main__':
    parser = ArgumentParser(
        description="DART Evaluation & Visualization scripts.")

    subparsers = parser.add_subparsers()
    for name, command in commands.items():
        p = subparsers.add_parser(
            name, help=command._desc, description=command._desc)
        command._parse(p)
        p.set_defaults(_func=command._main)

    args = parser.parse_args()
    args._func(args)
