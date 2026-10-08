"""
Step: word-group boundary detection, v0 -- simplest possible rules.

Goal: given a Hindi sentence, mark which tokens belong to the same
"word group" (e.g. verb + auxiliary chains, noun + postposition).

This is NOT meant to be linguistically perfect yet. It's meant to be
something you can look at and judge: does this roughly match what a
human would group together? Run on ~20-30 sentences and eyeball it
before trusting it on the full corpus.

No GPU needed -- safe to run while training is happening.
"""

# Common Hindi auxiliary/helping verbs that attach to a main verb
# to form one semantic unit (tense, aspect, mood markers).
AUXILIARIES = {
    "है", "हैं", "था", "थी", "थे", "हो", "होगा", "होगी", "होंगे",
    "रहा", "रही", "रहे", "गया", "गई", "गए", "सकता", "सकती", "सकते",
    "चुका", "चुकी", "चुके", "लगा", "लगी", "लगे", "पड़ा", "पड़ी", "पड़े",
    "दिया", "दी", "दिए", "लिया", "ली", "लिए", "डाला", "डाली", "डाले",
    "जाता", "जाती", "जाते", "करता", "करती", "करते", "वाला", "वाली", "वाले",
}

# Common postpositions/case markers that attach to the preceding noun.
POSTPOSITIONS = {
    "का", "की", "के", "को", "से", "में", "पर", "ने", "तक", "भी", "ही",
    "साथ", "पास", "ऊपर", "नीचे", "बीच", "नहीं",
}

# Compound postpositions -- these are multi-word units themselves
# (e.g. "के लिए" = "for"). A single-word match won't catch these;
# we check for them as adjacent word pairs separately.
COMPOUND_POSTPOSITIONS = {
    ("के", "लिए"), ("के", "साथ"), ("के", "बाद"), ("के", "पास"),
    ("के", "बारे"), ("की", "वजह"), ("के", "कारण"),
}


def find_word_groups(sentence: str):
    """
    Very simple rule: a token joins the PREVIOUS group if it's an
    auxiliary or postposition. Otherwise it starts a new group.

    Returns a list of groups, where each group is a list of words.
    """
    words = sentence.strip().split()
    groups = []
    current_group = []

    for word in words:
        # Strip trailing punctuation for the lookup, but keep it in the word
        clean_word = word.strip("।,.!?")

        if clean_word in AUXILIARIES or clean_word in POSTPOSITIONS:
            # Attach to the current group rather than starting a new one
            if current_group:
                current_group.append(word)
            else:
                current_group = [word]
        else:
            # Start a new group, closing off the previous one
            if current_group:
                groups.append(current_group)
            current_group = [word]

    if current_group:
        groups.append(current_group)

    return groups


def print_groups(sentence: str):
    groups = find_word_groups(sentence)
    print(f"\nSentence: {sentence}")
    print("Groups:", " | ".join(" ".join(g) for g in groups))


if __name__ == "__main__":
    # Hand-picked test sentences -- mix of simple and verb-chain-heavy cases
    test_sentences = [
        "भारत एक विशाल और विविधतापूर्ण देश है।",
        "मैं कल बाजार जा रहा था।",
        "उसने किताब पढ़ी।",
        "वह स्कूल जा चुका है।",
        "मुझे यह काम करना है।",
        "बच्चे पार्क में खेल रहे हैं।",
        "यह मेरी किताब है।",
        "वह घर से बाहर गया।",
    ]

    for sent in test_sentences:
        print_groups(sent)

    print("\n--- Now judge by eye ---")
    print("Does 'जा रहा था' get grouped together in sentence 2?")
    print("Does 'जा चुका है' get grouped together in sentence 4?")
    print("Do noun + postposition pairs (घर से, पार्क में) look right?")