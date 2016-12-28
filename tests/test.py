import re

liste = [',', '=', 'inc.', 'INC.']

for l in liste:
    reg = re.match("[a-zA-Z]", l)
    if l.isupper() and reg:
        print(l)