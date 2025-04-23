import re

pattern = r'(\w+), (\w+): (\d+\.\d+)'
text = ''
with open('similarities.txt', 'r') as file:
    text = file.read()

matches = re.findall(pattern, text)

langs = set()
langs.update([m[0] for m in matches])
langs.update([m[1] for m in matches])
print(len(langs))
print(langs)

with open('new.txt', 'w') as file:
    for match in matches:
        file.write(f'{match[0]}, {match[1]}: {match[2]}\n')