#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Dus'li

set -euo pipefail

TOPDIR=$(git rev-parse --show-toplevel)
FIND_FLAGS="-regextype posix-extended"

for file in $(find "${TOPDIR}" ${FIND_FLAGS} -regex '.*\.(cu|cxx|cuh|hxx|tpp)'); do
	clang-format --style=file -i "${file}"
done
