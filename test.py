word = 'Saint-Saëns'
word = '°'
print(word)
print(word.isupper())
print(word.islower())
print(word.isdigit())
print(word[0].isupper() and word[1:].islower())
print(word.istitle())
print(word.lower() == word.upper())
print(word.lower())
print(word.lower().upper())
print(word.encode('ascii', 'ignore').decode("utf-8") )


