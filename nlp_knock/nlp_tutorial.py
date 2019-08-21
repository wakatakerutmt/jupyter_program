import math
import sys

n = int(sys.argv[1])

lines = list(sys.stdin)
size = math.ceil(len(lines) / n)  # math.ceil は小数点以下を切り上げてくれます

for i in range(n):
    filename = 'nlp16.{0:02}'.format(i)
    with open(filename, 'w') as f:
        for l in lines[i * size: (i + 1) * size]:
            f.write(l)
