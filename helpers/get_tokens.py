import re

with open('./text_preprocessing.py', 'r', encoding='utf-8') as f:
	lines = f.readlines()
	
pattern = r'\[[A-Z_]+\]'
tags = [ re.findall(pattern, line) for line in lines ]
tags = [ tag for tag in tags if len(tag) > 0 ]
tags = [ t for tag in tags for t in tag ]
sorted_tags = set(tags)

'''
임시 태그
[DAYS]
[POINT_COM]
[POINT_DOT]
'''

temp_tags = ['[DAYS]','[POINT_COM]', '[POINT_DOT]']

filtered_tags = sorted(set(sorted_tags) - set(temp_tags))
for tag in filtered_tags:
	print(tag)