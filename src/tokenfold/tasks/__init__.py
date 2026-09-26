"""Public datasets reshaped into tokenfold rows: {task, heldout, state, question, label}.

The point is breadth of question *shapes*, not any one domain: NLI and paraphrase become `noul`,
topic/intent/emotion become `choice`, rating scales become `score`. Each Task fixes the question
(instructions + criteria) and a `convert(row, names)` that returns (state, label), or None to skip a row.

`heldout` tasks are never used to fit calibration (and later, never trained on). They measure whether
what we fit on the other tasks transfers to question shapes the model has not seen, which is the
production situation: callers define new questions at request time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_CHARS = 1500  # keeps every prompt well inside a 4k context


def clip(s: str) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS] + " …"


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    config: str
    split: str
    type: str  # choice | score | noul
    instructions: str
    convert: Callable[[dict, list[str] | None], tuple[Any, Any] | None]
    criteria: Any = None  # dict (choice), list (score), optional dict (noul); or callable(names) -> dict
    label_field: str = "label"  # ClassLabel whose names are passed to convert/criteria
    expect_names: tuple[str, ...] | None = None  # guard: fail loudly if the dataset's label order changed
    heldout: bool = False
    n: int | None = None  # per-task sample cap (e.g. many-option tasks are slow to score)
    page: int = 20  # rows per fetched page; smaller for label-sorted splits so samples span labels

    def question(self, names: list[str] | None) -> dict:
        crit = self.criteria(names) if callable(self.criteria) else self.criteria
        q = {"type": self.type, "instructions": self.instructions}
        if crit is not None:
            q["criteria"] = crit
        return q


def by_index(keys: list[str], field: str = "label", **state_fields: str):
    """convert() for ClassLabel tasks: state = {state_key: row[row_field]}, label = keys[row[field]]."""

    def convert(row: dict, _names):
        y = row.get(field)
        if y is None or not (0 <= y < len(keys)):
            return None  # GLUE test splits carry -1
        return {k: clip(row[f]) for k, f in state_fields.items()}, keys[y]

    return convert


def yes_if(value, field: str = "label", **state_fields: str):
    """convert() for noul tasks: label is True when row[field] == value."""

    def convert(row: dict, _names):
        y = row.get(field)
        if y is None or y == -1:
            return None
        return {k: clip(row[f]) for k, f in state_fields.items()}, y == value

    return convert


def _stsb(row: dict, _names):
    state = {"sentence1": clip(row["sentence1"]), "sentence2": clip(row["sentence2"])}
    return state, round(row["label"])


def _yahoo(row: dict, names):
    keys = [k for k, _ in YAHOO]
    state = {"title": clip(row["question_title"]), "body": clip(row["question_content"] or "")}
    return state, keys[row["topic"]]


def _clinc(row: dict, names):
    return {"utterance": clip(row["text"])}, names[row["intent"]]


def _clinc_criteria(names):
    return {n: ("out of scope: none of the other intents apply" if n == "oos" else None) for n in names}


def _banking_key(name: str) -> str:
    return name.lower().rstrip("?")  # upstream has "Refund_not_showing_up" and "reverted_card_payment?"


def _banking(row: dict, names):
    return {"message": clip(row["text"])}, _banking_key(names[row["label"]])


def _banking_criteria(names):
    return {_banking_key(n): None for n in names}


YAHOO = [
    ("society_culture", "Society & Culture"),
    ("science_math", "Science & Mathematics"),
    ("health", "Health"),
    ("education", "Education & Reference"),
    ("computers_internet", "Computers & Internet"),
    ("sports", "Sports"),
    ("business_finance", "Business & Finance"),
    ("entertainment_music", "Entertainment & Music"),
    ("family_relationships", "Family & Relationships"),
    ("politics_government", "Politics & Government"),
]

DBPEDIA = [
    ("company", "a company or business"),
    ("school", "an educational institution"),
    ("artist", "an artist, musician, or writer"),
    ("athlete", "an athlete"),
    ("politician", "an office holder or politician"),
    ("vehicle", "a means of transportation (vehicle, ship, aircraft)"),
    ("building", "a building or structure"),
    ("natural_place", "a natural place (mountain, river, lake)"),
    ("village", "a village or small settlement"),
    ("animal", "an animal species"),
    ("plant", "a plant species"),
    ("album", "a music album"),
    ("film", "a film"),
    ("book", "a written work (book, journal, magazine)"),
]

STARS = [
    "1 star: terrible",
    "2 stars: poor",
    "3 stars: average",
    "4 stars: good",
    "5 stars: excellent",
]
NLI_3 = {
    "entailment": "the hypothesis must be true if the premise is true",
    "neutral": "the hypothesis might or might not be true given the premise",
    "contradiction": "the hypothesis cannot be true if the premise is true",
}

TASKS: list[Task] = [
    # ---- choice ----------------------------------------------------------------------------------
    Task(
        "ag_news",
        "fancyzhx/ag_news",
        "default",
        "test",
        "choice",
        "What is the topic of `article`?",
        by_index(["world", "sports", "business", "scitech"], article="text"),
        criteria={
            "world": "World news, politics, international affairs",
            "sports": "Sports",
            "business": "Business, economy, markets",
            "scitech": "Science and technology",
        },
        expect_names=("World", "Sports", "Business", "Sci/Tech"),
    ),
    Task(
        "emotion",
        "dair-ai/emotion",
        "split",
        "test",
        "choice",
        "Which emotion does `message` primarily express?",
        by_index(["sadness", "joy", "love", "anger", "fear", "surprise"], message="text"),
        criteria={
            "sadness": None,
            "joy": None,
            "love": "love or affection",
            "anger": None,
            "fear": "fear or anxiety",
            "surprise": None,
        },
        expect_names=("sadness", "joy", "love", "anger", "fear", "surprise"),
    ),
    Task(
        "mnli",
        "nyu-mll/glue",
        "mnli",
        "validation_matched",
        "choice",
        "What is the logical relationship between `premise` and `hypothesis`?",
        by_index(list(NLI_3), premise="premise", hypothesis="hypothesis"),
        criteria=NLI_3,
        expect_names=("entailment", "neutral", "contradiction"),
    ),
    Task(
        "tweet_sentiment",
        "cardiffnlp/tweet_eval",
        "sentiment",
        "test",
        "choice",
        "What sentiment does `tweet` express?",
        by_index(["negative", "neutral", "positive"], tweet="text"),
        criteria={"negative": None, "neutral": None, "positive": None},
        expect_names=("negative", "neutral", "positive"),
    ),
    Task(
        "yahoo_topics",
        "community-datasets/yahoo_answers_topics",
        "yahoo_answers_topics",
        "test",
        "choice",
        "Which category does the question with `title` and `body` belong to?",
        _yahoo,
        criteria=dict(YAHOO),
        label_field="topic",
        expect_names=tuple(v for _, v in YAHOO),
    ),
    Task(
        "banking77",
        "legacy-datasets/banking77",
        "default",
        "test",
        "choice",
        "Which intent best matches the bank customer's `message`?",
        _banking,
        criteria=_banking_criteria,
        n=60,  # 77 candidates per question, like clinc
        page=1,  # test split is sorted by label
    ),
    Task(
        "yelp_polarity",
        "fancyzhx/yelp_polarity",
        "plain_text",
        "test",
        "choice",
        "Is the overall sentiment of `review` negative or positive?",
        by_index(["negative", "positive"], review="text"),
        criteria={"negative": None, "positive": None},
        expect_names=("1", "2"),
    ),
    Task(
        "dbpedia",
        "fancyzhx/dbpedia_14",
        "dbpedia_14",
        "test",
        "choice",
        "What kind of entity is described by `title` and `content`?",
        by_index([k for k, _ in DBPEDIA], title="title", content="content"),
        criteria=dict(DBPEDIA),
        heldout=True,
        expect_names=(
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ),
    ),
    Task(
        "clinc_oos",
        "clinc/clinc_oos",
        "plus",
        "test",
        "choice",
        "Which intent best matches `utterance`?",
        _clinc,
        criteria=_clinc_criteria,
        label_field="intent",
        heldout=True,
        n=60,
        page=1,
    ),
    Task(
        "subj",
        "SetFit/subj",
        "default",
        "test",
        "choice",
        "Is `sentence` a subjective opinion or an objective statement of fact?",
        by_index(["objective", "subjective"], sentence="text"),
        criteria={
            "objective": "states facts or describes events",
            "subjective": "expresses opinion or feeling",
        },
        heldout=True,
    ),
    # ---- score -----------------------------------------------------------------------------------
    Task(
        "sst5",
        "SetFit/sst5",
        "default",
        "test",
        "score",
        "How positive is the sentiment of `review`?",
        lambda row, _: ({"review": clip(row["text"])}, int(row["label"])),
        criteria=["very negative", "negative", "neutral or mixed", "positive", "very positive"],
    ),
    Task(
        "yelp",
        "Yelp/yelp_review_full",
        "yelp_review_full",
        "test",
        "score",
        "How many stars would the author of `review` give?",
        lambda row, _: ({"review": clip(row["text"])}, int(row["label"])),
        criteria=STARS,
    ),
    Task(
        "amazon",
        "mteb/amazon_reviews_multi",
        "en",
        "test",
        "score",
        "Rate the customer's overall satisfaction expressed in `review`.",
        lambda row, _: ({"review": clip(row["text"])}, int(row["label"])),
        criteria=STARS,
        heldout=True,
        page=2,
    ),
    Task(
        "stsb",
        "nyu-mll/glue",
        "stsb",
        "validation",
        "score",
        "How similar in meaning are `sentence1` and `sentence2`?",
        _stsb,
        heldout=True,
        criteria=[
            "completely dissimilar",
            "not equivalent, but on the same topic",
            "not equivalent, but share some details",
            "roughly equivalent, but some important information differs",
            "mostly equivalent, only minor details differ",
            "completely equivalent",
        ],
    ),
    # ---- noul ------------------------------------------------------------------------------------
    Task(
        "rte",
        "nyu-mll/glue",
        "rte",
        "validation",
        "noul",
        "Does `premise` imply that `hypothesis` is true?",
        yes_if(0, premise="sentence1", hypothesis="sentence2"),
        criteria={"true": "the hypothesis follows from the premise", "false": "it does not follow"},
        expect_names=("entailment", "not_entailment"),
    ),
    Task(
        "mrpc",
        "nyu-mll/glue",
        "mrpc",
        "validation",
        "noul",
        "Do `sentence1` and `sentence2` say the same thing?",
        yes_if(1, sentence1="sentence1", sentence2="sentence2"),
        criteria={"true": "paraphrases of each other", "false": "different meaning"},
        expect_names=("not_equivalent", "equivalent"),
    ),
    Task(
        "qqp",
        "nyu-mll/glue",
        "qqp",
        "validation",
        "noul",
        "Are `question1` and `question2` asking the same thing?",
        yes_if(1, question1="question1", question2="question2"),
        expect_names=("not_duplicate", "duplicate"),
    ),
    Task(
        "paws",
        "google-research-datasets/paws",
        "labeled_final",
        "test",
        "noul",
        "Do `sentence1` and `sentence2` have the same meaning?",
        yes_if(1, sentence1="sentence1", sentence2="sentence2"),
        criteria={"true": "paraphrases", "false": "different meaning, even if the words overlap"},
        heldout=True,
    ),
    Task(
        "boolq",
        "google/boolq",
        "default",
        "validation",
        "noul",
        "Based on `passage`, is the answer to `question` yes?",
        yes_if(True, field="answer", passage="passage", question="question"),
    ),
    Task(
        "scitail",
        "allenai/scitail",
        "snli_format",
        "test",
        "noul",
        "Does `premise` support `hypothesis`?",
        yes_if("entailment", field="gold_label", premise="sentence1", hypothesis="sentence2"),
        heldout=True,
    ),
    Task(
        "tweet_offensive",
        "cardiffnlp/tweet_eval",
        "offensive",
        "test",
        "noul",
        "Is `tweet` offensive?",
        yes_if(1, tweet="text"),
        criteria={"true": "insults, profanity, or targeted offense", "false": "not offensive"},
        expect_names=("non-offensive", "offensive"),
    ),
    Task(
        "tweet_hate",
        "cardiffnlp/tweet_eval",
        "hate",
        "test",
        "noul",
        "Does `tweet` contain hate speech against immigrants or women?",
        yes_if(1, tweet="text"),
        expect_names=("non-hate", "hate"),
    ),
    Task(
        "tweet_irony",
        "cardiffnlp/tweet_eval",
        "irony",
        "test",
        "noul",
        "Is `tweet` ironic or sarcastic?",
        yes_if(1, tweet="text"),
        expect_names=("non_irony", "irony"),
    ),
    Task(
        "cola",
        "nyu-mll/glue",
        "cola",
        "validation",
        "noul",
        "Is `sentence` grammatically acceptable English?",
        yes_if(1, sentence="sentence"),
        expect_names=("unacceptable", "acceptable"),
    ),
    Task(
        "sms_spam",
        "ucirvine/sms_spam",
        "plain_text",
        "train",
        "noul",
        "Is `sms` spam?",
        yes_if(1, sms="sms"),
        criteria={
            "true": "unsolicited advertising, scams, or prize offers",
            "false": "a normal personal message",
        },
        expect_names=("ham", "spam"),
    ),
    Task(
        "sst2",
        "stanfordnlp/sst2",
        "default",
        "validation",
        "noul",
        "Is `review` positive?",
        yes_if(1, review="sentence"),
        expect_names=("negative", "positive"),
    ),
]

BY_NAME = {t.name: t for t in TASKS}
