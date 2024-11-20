import sys

# 💥💥💥 value = yield from EXPR
def accumulate():
    tally = 0
    while 1:
        next = yield
        if next == 'finish':
            return tally
        tally += next

def gather_tallies(tallies):
    while 1:
        tally = yield from accumulate()
        tallies.append(tally)

tallies = []
acc = gather_tallies(tallies)
next(acc)  # Ensure the accumulator is ready to accept values

for i in range(4):
    acc.send(i)
acc.send('finish')  # Finish the first tally

for i in range(5):
    acc.send(i)
acc.send('finish')  # Finish the second tally

print(tallies)

sys.exit()


# 💥💥💥 Test next(), send(), throw() and close()
def echo(value=None):
    print("Execution starts when 'next()' is called for the first time.")
    try:
        while True:
            try:
                value = (yield value)
            except Exception as e:
                value = e
    finally:
        print("Don't forget to clean up when 'close()' is called.")

generator = echo(1)

print(next(generator)) # 1

print(next(generator)) # None

print(next(generator)) # None

print(generator.send(2)) # 2

print(generator.throw(TypeError, "spam")) # spam

generator.close()


# 💥💥💥 Simple generator: yield values one by one
# This function pauses to return a value to the outer scope, then proceeds
DICTIONARY = {
    'a': 'apple',
    'b': 'banana',
    'c': 'cat',
    'd': 'dog',
    'e': 'egoseismic',
    'f': 'facepalm',
    'g': 'gringo',
    'h': 'hilarious',
    'i': 'iPhone',
    'j': 'John Lennon',
    'k': 'KDE',
    'l': 'lemonsaurus',
    'm': 'Mickey Mouse',
    'n': 'Netherrealm',
    'o': 'of course',
    'p': 'pokerface',
}

# 💥💥💥 A generator that uses `yield` to request data from the outside
# This use case, however, is a little inside-out. Makes very little sense. But you can :)

def gimme_words():
    # Start building a sentence
    sentence = []
    for letter in 'abcdef':
        # Give every letter to some external caller and expect them to give us a word
        word = yield letter
        print(f"In generator: {word=}")
        # Keep building the sentence
        sentence.append(word)

    # Return the sentence!
    return ' '.join(sentence)


# Initialize the generator
g = gimme_words()

# Keep going while we can
try:
    word = None
    while True:
        # Send the initial `None` to the generator. We just have to.
        # When send() is called to start the generator, it must be called with 'None' as the
        #   argument, because there is no yield expression that could receive the value.
        letter = g.send(word)
        print(f"In main: {letter=}")
        # Convert the letter that the generator gives us into a word and give it back to the generator
        word = DICTIONARY[letter]
# When done, print the resulting value
except StopIteration as e:
    print(e.value)
finally:
    g.close()