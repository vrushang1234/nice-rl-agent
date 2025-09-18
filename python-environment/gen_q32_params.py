#!/usr/bin/env python3
import csv, sys, pathlib

# Network sizes (keep synced with nn_policy.h)
INPUT_SIZE = 6
H1 = 50
H2 = 70
OUT = 11

Q = 32
SCALE = 1 << Q
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1

def q32(x: float) -> int:
    v = int(round(x * SCALE))
    return I64_MIN if v < I64_MIN else I64_MAX if v > I64_MAX else v

def alloc():
    return {
        'w1': [[0]*INPUT_SIZE for _ in range(H1)],
        'b1': [0]*H1,
        'w2': [[0]*H1 for _ in range(H2)],
        'b2': [0]*H2,
        'w3': [[0]*H2 for _ in range(OUT)],
        'b3': [0]*OUT,
    }

def emit_matrix(name, M):
    rows = len(M); cols = len(M[0]) if rows else 0
    s = f"const q32_32 {name}[{rows}][{cols}] = {{\n"
    for r in range(rows):
        s += "    { " + ", ".join(str(M[r][c]) for c in range(cols)) + " },\n"
    s += "};\n\n"
    return s

def emit_vector(name, V):
    n = len(V)
    s = f"const q32_32 {name}[{n}] = {{\n    "
    line = []
    for i, v in enumerate(V):
        line.append(str(v))
        if (i+1) % 10 == 0 and (i+1) != n:
            s += ", ".join(line) + ",\n    "
            line = []
    if line: s += ", ".join(line)
    s += "\n};\n\n"
    return s

BIAS_ALIASES = {
    'b1','b_1','bias1','bias_1','layer1_bias','l1_bias',
    'b2','b_2','bias2','bias_2','layer2_bias','l2_bias',
    'b3','b_3','bias3','bias_3','layer3_bias','l3_bias',
}

def norm_param(p):
    p = p.strip().lower()
    if p in BIAS_ALIASES:
        if '1' in p: return 'b1'
        if '2' in p: return 'b2'
        if '3' in p: return 'b3'
    return p

def main(csv_path, out_path):
    T = alloc()
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            p = norm_param(row['param'])
            r = int(row['row'])
            v = q32(float(row['value']))
            if p in ('w1','w2','w3'):
                c = int(row['col'])
                if p == 'w1':
                    assert 0 <= r < H1 and 0 <= c < INPUT_SIZE
                    T['w1'][r][c] = v
                elif p == 'w2':
                    assert 0 <= r < H2 and 0 <= c < H1
                    T['w2'][r][c] = v
                else:
                    assert 0 <= r < OUT and 0 <= c < H2
                    T['w3'][r][c] = v
            elif p in ('b1','b2','b3'):
                # allow col to be missing or -1
                if p == 'b1':
                    assert 0 <= r < H1
                    T['b1'][r] = v
                elif p == 'b2':
                    assert 0 <= r < H2
                    T['b2'][r] = v
                else:
                    assert 0 <= r < OUT
                    T['b3'][r] = v
            else:
                raise ValueError(f"Unknown param '{row['param']}' (normalized='{p}')")

    src = []
    src.append('// Auto-generated from weights CSV -> Q32.32')
    src.append('#include "nn_policy.h"\n')
    src.append(emit_matrix('W1', T['w1']))
    src.append(emit_vector('B1', T['b1']))
    src.append(emit_matrix('W2', T['w2']))
    src.append(emit_vector('B2', T['b2']))
    src.append(emit_matrix('W3', T['w3']))
    src.append(emit_vector('B3', T['b3']))

    pathlib.Path(out_path).write_text("".join(src))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} weights.csv nn_params.c")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

