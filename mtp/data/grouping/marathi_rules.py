"""JI-7: Marathi whitespace groups; suffixes remain inside their original words.

Auxiliary/copular attachment is lexical, not a dependency parse. नाही before a
recognised verbal form starts a forward negation group; after a recognised verb it attaches
left; without a recognised host it stays separate. लागला and शकतो variants join a recognised
verb, avoiding arbitrary noun + main-verb attachments. च्या साठी etc. match as
windows; fused forms (घरात/घरासाठी) are never split. Lists need gold validation.
"""

from .hindi_rules import HindiRuleGrouperV1

WORD_LISTS = {
    "aux_chain": "आहे आहेत होता होती होते होतो होते होत्या होत्या असेल असतील आहेस आहोत आहात होईल होतील नाही नव्हता नव्हती नव्हते".split(),
    "light_verb": "लागला लागली लागले लागतो लागते लागतात".split(),
    "modal": "शकतो शकते शकतात शकला शकली शकले".split(),
    "postposition": "च्या ची चे ला ने शी साठी बद्दल नंतर आधी मध्ये वर खाली पासून पर्यंत जवळ कडे ही सुद्धा देखील".split(),
    "compound_postposition": [
        p.split()
        for p in [
            "च्या साठी",
            "च्या बद्दल",
            "च्या नंतर",
            "च्या आधी",
            "च्या मध्ये",
            "च्या जवळ",
            "च्या मुळे",
            "च्या वर",
            "च्या खाली",
            "च्या बरोबर",
            "च्या कडे",
        ]
    ],
    "verb_bases": "जा ये कर हो वाच लिही लिह बोल सांग ऐक बघ पाह खा पी दे घे ठेव चाल बस उठ झोप खेळ उघड समज शिक बनव मिळव पाठव परत भेट विकत विक मार काप धुव घाल निघ पोहोच राह पड फेक टाक लाग बदल थांब सोड भर निवड शोध माग विसर विचार पळ नाच गा जिंक हर तुट तोड जोड उड".split(),
    "verb_irregular": "केला केली केले करत करतो करते करतात करीत करणे करण्यास करताना गेला गेली गेले जात जातो जाते जातात जाणे जाणार आला आली आले येत येतो येते येतात येणे झाला झाली झाले होत होणे वाचत वाचतो वाचते वाचणे लिहित लिहिणे सांगितले दिला दिली दिले घेतला घेतली घेतले खाल्ले राहिला राहिली राहिले".split(),
}


class MarathiRuleGrouper(HindiRuleGrouperV1):
    name = "mr_rules_v1"
    negation = "नाही"

    def __init__(self, lang="mr", **kwargs):
        if lang != "mr":
            raise ValueError("mr_rules_v1 is Marathi-only")
        self.lang = lang
        self.aux = set(WORD_LISTS["aux_chain"]) | {"नाही"}
        self.light = set(WORD_LISTS["light_verb"])
        self.post = set(WORD_LISTS["postposition"])
        self.habitual = set(WORD_LISTS["modal"])
        self.verbs = set(WORD_LISTS["verb_bases"]) | set(WORD_LISTS["verb_irregular"])
        self.verbs.update(
            b + e
            for b in WORD_LISTS["verb_bases"]
            for e in (
                "णे",
                "तो",
                "ते",
                "तात",
                "त",
                "ला",
                "ली",
                "ले",
                "णार",
                "ायला",
                "ू",
                "ून",
                "ताना",
                "ण्यास",
            )
        )
        self.compounds = sorted(
            (tuple(p) for p in WORD_LISTS["compound_postposition"]),
            key=lambda p: (-len(p), p),
        )
