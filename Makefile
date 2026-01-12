# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Dus'li

TARGET := hamming-one

CXX  := g++
NVCC := nvcc

DIR_INCLUDE       := include
DIR_SOURCE        := source
DIR_SOURCE_HOST   := $(DIR_SOURCE)/host
DIR_SOURCE_DEVICE := $(DIR_SOURCE)/device
DIR_BUILD         := build

CXXFLAGS  := -O2 -Wall -Wextra -std=c++20 -I$(DIR_INCLUDE)
NVCCFLAGS := -O2 -I$(DIR_INCLUDE) --extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += --std c++20 --compiler-options '-fPIC -Wall -Wextra'
LDFLAGS   :=

V ?= 0
ifeq ($(V),1)
	Q :=
else
	Q := @
endif

# yeah, AHA you know what it is... bold and yellow bold and yellow bold and...
BOLD_AND_YELLOW := \033[1;33m
BOLD_AND_GREEN  := \033[1;32m
RESET           := \033[0m

# Pretty print a command execution unless verbose mode is on.
define do_cmd
	@printf "  $(BOLD_AND_YELLOW)%-5s$(RESET) %s\n" "$(1)" "$(2)"
	$(Q)$(3)
endef

CU_SRCS  := $(shell find $(DIR_SOURCE) -name '*.cu')
CXX_SRCS := $(shell find $(DIR_SOURCE) -name '*.cxx')

CU_OBJS  := $(patsubst %.cu,$(DIR_BUILD)/%.o,$(CU_SRCS))
CXX_OBJS := $(patsubst %.cxx,$(DIR_BUILD)/%.o,$(CXX_SRCS))
ALL_OBJS := $(CU_OBJS) $(CXX_OBJS)

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(call do_cmd,LD,$@,$(NVCC) $(LDFLAGS) -o $@ $^)
	@printf "\n$(BOLD_AND_GREEN)Build successful$(RESET)\n"

$(DIR_BUILD)/%.o: %.cu
	@test -d $(dir $@) || mkdir -p $(dir $@)
	$(call do_cmd,NVCC,$<,$(NVCC) $(NVCCFLAGS) -c -o $@ $<)

$(DIR_BUILD)/%.o: %.cxx
	@test -d $(dir $@) || mkdir -p $(dir $@)
	$(call do_cmd,CXX,$<,$(CXX) $(CXXFLAGS) -c -o $@ $<)

clean:
	$(call do_cmd,RM,$(TARGET),rm -f $(TARGET))

distclean: clean
	$(call do_cmd,RM,$(DIR_BUILD),rm -rf $(DIR_BUILD))

.PHONY: all clean distclean
