<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Dus'li -->

# Hamming One

## Context

This is a second project realized for a CUDA programming uni class I'm attending.
The goal is to take a collection of binary words as input and locate all pairs
whose Hamming distance is exactly equal to one.

The solution is based on radix sorting with single-bit masking, which allows for
clustering H1 candidates close to each other.

## Running the app

If supplied with a single positional argument, the command inteprets it as path
to an input file, and attempts to read from it. Otherwise it reads from STDIN.

This means, that given only what is in the repository, following invocations all
should work:

``` sh
# Easy to verify the example just by looking at it.
./hamming-one examples/simple.txt

# Large amount of different words. First two only differ at a single position.
zcat examples/large.txt.gz | ./hamming-one

# It's not gambling, I can stop whenever I want! (random examples)
./scripts/mkexample.py LENGTH COUNT | ./hamming-one
```

## Navigating the project

``` text
├ 📁 LICENSES    Licenses
├ 📁 examples    Example inputs
├ 📁 include     Header files
├ 📁 scripts
│ └ 📁 hooks     Prevents me from pushing something stupid to remote
├ 📁 source      Source files
├ .clang-format  Aids in cleaning up the formatting
└ Makefile       Facilitates building the thing
```

## What is needed to build this

Apart from the CUDA toolchain nothing out of ordinary comes to mind.

I have only tested this on a GNU/Linux machine. Thanks to AI being a thing now,
it turns out that NVIDIA compatibility got way better than I remembered it from
last time. I don't recall doing anything Linux-specific in code, but the
Makefile will not work for Windows, unless its ran from like a WSL or Cygwin or
whatnot.
