// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include "compiler.cuh"

#define ARRAY_SIZE(_arr)                                       \
	({                                                     \
		_Static_assert(!same_type((_arr), &(_arr)[0]), \
		    "'" STRINGIFY(_arr) "' must be an array"); \
		sizeof(_arr) / sizeof(_arr[0]);                \
	})
