// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#define STRINGIFY__(token_) #token_
#define STRINGIFY(token_)   STRINGIFY__(token_)

#define boolify(val_) (!!(val_))

#define unlikely(pred_) __builtin_expect(boolify(pred_), 0)
#define likely(pred_)   __builtin_expect(boolify(pred_), 1)

#define same_type__(fst_, snd_) __builtin_types_compatible_p(fst_, snd_)
#define same_type(fst_, snd_)   same_type__(typeof(fst_), typeof(snd_))

#define fallthrough __attribute__((fallthrough))
