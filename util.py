# -*- coding: utf-8 -*-

##############################################################
# util.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


def erase_new_line(s):
    return s.strip()


def read_text(filename):
    with open(filename, 'r') as f:
        lines = map(erase_new_line, f.readlines())
    return lines


def chunk_list(input_list, n):
    return [input_list[x:x + n] for x in range(0, len(input_list), n)]
