"""Fix RereDiffTensor field references after privatization."""
import re, sys

MAPPING = {
    # field_name: (getter, setter)
    'value':              ('value()',              'setValue'),
    'grad':               ('gradData()',           'setGradData'),
    'inputs':             ('inputs()',             'setInputs'),
    'backwardFn':         ('backwardFn()',         'setBackwardFn'),
    'opTag':              ('opTag()',              'setOpTag'),
    'requiresGrad':       ('requiresGrad()',       'setRequiresGrad'),
    'isLeaf':             ('isLeaf()',             'setIsLeaf'),
    'scalarParam':        ('scalarParam()',        'setScalarParam'),
    'scalarParam2':       ('scalarParam2()',       'setScalarParam2'),
    'exportShape':        ('exportShape()',        'setExportShape'),
    'backwardIndices':    ('backwardIndices()',    'setBackwardIndices'),
    'symbolicBackwardFn': ('symbolicBackwardFn()', 'setSymbolicBackwardFn'),
}

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    orig = content

    for field, (getter, setter) in MAPPING.items():
        # Step 1: Fix ASSIGNMENTS: .field = VAL;
        # Match .field = up to the closing ; (non-greedy on the value part)
        pat_assign = r'\.' + field + r'\s*=\s*'
        def repl_assign(m):
            # Find the full assignment: = <value>;
            start = m.end()
            depth = 0
            i = start
            while i < len(content):
                c = content[i]
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                elif c == ';' and depth == 0:
                    val = content[start:i]
                    return '.' + setter + '(' + val + ');'
                i += 1
            return m.group(0)  # shouldn't happen
        # Actually, regex-based approach is simpler for most cases
        # Simple pattern: .field = <non-semicolon> ;
        pat_simple = re.compile(r'\.' + field + r'\s*=\s*([^;]+);')
        content = pat_simple.sub(r'.' + setter + r'(\1);', content)

        # Step 2: Fix READS: .field (not followed by = or ()
        # Must be followed by non-alphanumeric, non-(, non-=, non-space-then-(
        # .field followed by: space, comma, semicolon, ), ], +, -, *, /, &, |, >, <, !, ?, :
        # But NOT followed by ( or =
        pat_read = re.compile(r'\.' + field + r'(?!\s*[=\(])')
        content = pat_read.sub('.' + getter, content)

    if content != orig:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

if __name__ == '__main__':
    files = sys.argv[1:]
    for f in files:
        if fix_file(f):
            print(f'Fixed: {f}')
        else:
            print(f'No changes: {f}')
