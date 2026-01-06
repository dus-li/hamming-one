<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Dus'li -->

# Hamming One

## Context

TODO

## Navigating the project

``` text
├ 📁 LICENSES    Licenses
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
