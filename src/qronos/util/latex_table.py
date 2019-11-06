#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

def generate_table(columns, systems):
    '''
    print as a table:
    rows: every system in systems
    columns: defined by tuples (title, alignment, generator_function(system)) in columns
    '''
    latex = r'\begin{tabular}{' + (''.join([i[1] for i in columns])) + r'}\hline' + '\n'
    latex += " & ".join([title for (title, _, _) in columns]) + r"\\ \hline" + "\n"
    def generate_row(system, columns):
        return " & ".join([str(generate_content(system)) for (_, _, generate_content) in columns])
    latex += "\\\\\n".join([generate_row(system, columns) for system in systems])
    latex += "\\\\\\hline\n\\end{tabular}"
    return latex

def format_float_ceil(number, digits):
    """
    format floating-point value to given number of decimal places, round up last digit
    >>> format_float_ceil(1.8,1)
    '$1.8$'
    >>> format_float_ceil(1.81,1)
    '$1.9$'
    >>> format_float_ceil(1.800001,1)
    '$1.9$'
    """
    if number is None:
        return '---'
    if number == float('inf'):
        return '$\infty$'
    if number == -float('inf'):
        return '$-\infty$'
    return ("${:." + str(digits) + "f}$").format(math.ceil(number * (10 ** digits))/(10. ** digits))


def format_float_sci_ceil(number, digits):
    '''
    format floating-point value in scientific notation,
    with given given number of decimal places,
    round up last digit.
    
    >>> format_float_sci_ceil(12345.67890, 3)
    '1.235 \\cdot 10^{4}'
    '''
    exponent = math.floor(math.log10(number))
    return ("${:." + str(digits) + "f} \cdot 10^{{{:.0f}}}$").format(math.ceil(number / (10 ** exponent) * (10 ** digits))/(10. ** digits), exponent)