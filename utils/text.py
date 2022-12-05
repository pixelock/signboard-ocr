# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : text.py
@Date    : 2022/11/2 23:46 
"""

import zhconv


def simplified2traditional(text):
    """Transfer text from simplified chinese to traditional chinese.
    """
    return zhconv.convert(text, 'zh-hant')


def traditional2simplified(text):
    """Transfer text from traditional chinese to simplified chinese.
    """
    return zhconv.convert(text, 'zh-hans')


def sbc2dbc_char(char):
    """Transfer single char from SBC case(半角) to DBC case(全角)
    """
    inside_code = ord(char)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return char
    elif inside_code == 0x0020:  # 0x0020为半角空格. 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0

    return chr(inside_code)


def dbc2sbc_char(char):
    """Transfer single char from DBC case(全角) to SBC case(半角)
    """
    inside_code = ord(char)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return char
    return chr(inside_code)


def sbc2dbc(text):
    """Transfer text from SBC case(半角) to DBC case(全角)
    """
    return ''.join([sbc2dbc_char(char) for char in text])


def dbc2sbc(text):
    """Transfer text from DBC case(全角) to SBC case(半角)
    """
    return ''.join([dbc2sbc_char(char) for char in text])


def is_chinese_char(char):
    return '\u4e00' <= char <= '\u9fa5'


def is_number_char(char):
    if '\u0030' <= char <= '\u0039':  # 半角数字
        return True
    elif '\uff10' <= char <= '\uff19':  # 全角数字
        return True
    else:
        return False


def is_alpha_char(char):
    if '\u0041' <= char <= '\u005a':  # 半角大写字母
        return True
    elif '\u0061' <= char <= '\u007a':  # 半角小写字母
        return True
    elif '\uff21' <= char <= '\uff3a':  # 全角大写字母
        return True
    elif '\uff41' <= char <= '\uff5a':  # 全角小写字母
        return True
    else:
        return False


def is_lower_char(char):
    if '\u0061' <= char <= '\u007a':  # 半角小写字母
        return True
    elif '\uff41' <= char <= '\uff5a':  # 全角小写字母
        return True
    else:
        return False


def is_upper_char(char):
    if '\u0041' <= char <= '\u005a':  # 半角大写字母
        return True
    elif '\uff21' <= char <= '\uff3a':  # 全角大写字母
        return True
    else:
        return False
