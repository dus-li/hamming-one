#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Dus'li

import argparse
import random
import sys

PARSER = argparse.ArgumentParser(description='Generate binary words')
PARSER.add_argument('-o', '--output', type=str, nargs='?',
                    help='output file path')
PARSER.add_argument('length', type=int,
                    help='length of single word')
PARSER.add_argument('count', type=int,
                    help='number of words to generate')


def generate(length, count, out) -> [str]:
    print(length, file=out)
    print(count, file=out)
    for _ in range(count):
        word = ''.join(random.choices('01', k=length))
        print(word, file=out)


def main():
    args = PARSER.parse_args()
    out = sys.stdout if args.output is None else open(args.output, 'w')
    generate(args.length, args.count, out)
    if args.output is not None:
        out.close()


if __name__ == "__main__":
    main()
