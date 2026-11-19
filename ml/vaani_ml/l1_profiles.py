"""L1-aware feedback for Indian-English speakers (Telugu and Hindi first).

Each entry describes a contrast that is commonly reported for speakers of
these languages learning General American English. `prior` (0-1) is how much
extra practice weight the scheduler gives the phoneme before it has any data
for the user; after a few attempts the user's own scores dominate.

These are coaching heuristics, not claims about any individual speaker.
Indian English is a valid variety; the app targets General American only
because that is what the reference model and Azure en-US assess against.
"""
from __future__ import annotations

CONTRASTS = {
    "V": {"often_heard_as": ["W", "B"], "prior": 0.9,
          "tip": "Rest your upper teeth lightly on your lower lip and keep the sound buzzing. "
                 "For W, round your lips and keep your teeth away from them."},
    "W": {"often_heard_as": ["V"], "prior": 0.8,
          "tip": "Start with rounded lips, like 'oo', then glide into the vowel. No teeth on the lip."},
    "Z": {"often_heard_as": ["JH", "S"], "prior": 0.8,
          "tip": "Keep the tongue tip behind the upper teeth like S, and add voice. Don't touch the "
                 "roof of the mouth; that makes J (dʒ)."},
    "TH": {"often_heard_as": ["T", "S"], "prior": 0.85,
           "tip": "Put the tongue tip lightly between your teeth and blow air through. Don't stop the air like T."},
    "DH": {"often_heard_as": ["D", "Z"], "prior": 0.85,
           "tip": "Same tongue position as 'th' in 'think', but with voice. Feel the buzz on the tongue tip."},
    "AE": {"often_heard_as": ["EH", "AA"], "prior": 0.7,
           "tip": "Open the jaw wider than for 'e' in 'bed', and spread the lips: 'cat', 'bad', 'man'."},
    "IH": {"often_heard_as": ["IY"], "prior": 0.6,
           "tip": "Short and relaxed: 'ship', 'bit'. Don't stretch it into 'ee' as in 'sheep'."},
    "IY": {"often_heard_as": ["IH"], "prior": 0.5,
           "tip": "Long and tense, with spread lips: 'sheep', 'beat', 'leave'."},
    "UH": {"often_heard_as": ["UW"], "prior": 0.55,
           "tip": "Short and relaxed: 'book', 'full'. Not the long 'oo' of 'fool'."},
    "OW": {"often_heard_as": ["AO"], "prior": 0.6,
           "tip": "It's a glide, not a single vowel: start at 'o' and move to 'w' – 'go', 'home'."},
    "EY": {"often_heard_as": ["EH"], "prior": 0.55,
           "tip": "Glide from 'e' to 'ee': 'day', 'late', 'make'."},
    "SH": {"often_heard_as": ["S"], "prior": 0.5,
           "tip": "Pull the tongue slightly back and round your lips: 'she', 'nation'."},
    "ZH": {"often_heard_as": ["JH", "SH"], "prior": 0.7,
           "tip": "The voiced partner of SH, as in 'measure' and 'vision'. Keep it smooth, no 'j' stop."},
    "F": {"often_heard_as": ["P"], "prior": 0.45,
          "tip": "Upper teeth on lower lip and let air flow continuously. Don't close the lips like P."},
    "ER": {"often_heard_as": ["AH"], "prior": 0.5,
           "tip": "Curl or bunch the tongue and keep the R colour through the vowel: 'bird', 'work'."},
    "AH": {"often_heard_as": ["AA"], "prior": 0.4,
           "tip": "Unstressed syllables usually reduce to a quick, relaxed 'uh' (schwa): 'about', 'banana'."},
    "S": {"often_heard_as": ["SH"], "prior": 0.35,
          "tip": "At the start of words like 'school' or 'speak', start straight with S. Don't add a vowel before it."},
}

DEFAULT_PRIOR = 0.25

PROFILES = {
    "te": {"name": "Telugu", "boost": {"V": 0.1, "W": 0.1, "Z": 0.05, "F": 0.1}},
    "hi": {"name": "Hindi", "boost": {"V": 0.1, "W": 0.1, "S": 0.1, "ZH": 0.05}},
    "other": {"name": "Other / general", "boost": {}},
}


def prior_for(phone: str, l1: str = "other") -> float:
    base = CONTRASTS.get(phone, {}).get("prior", DEFAULT_PRIOR)
    return min(1.0, base + PROFILES.get(l1, PROFILES["other"])["boost"].get(phone, 0.0))


def feedback_for(phone: str, heard: str | None) -> dict | None:
    c = CONTRASTS.get(phone)
    if not c:
        return None
    return {"phone": phone, "heard": heard, "tip": c["tip"],
            "known_confusion": bool(heard and heard in c["often_heard_as"])}
