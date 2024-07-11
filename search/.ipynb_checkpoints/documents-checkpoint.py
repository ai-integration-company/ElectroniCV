from collections import Counter
from dataclasses import dataclass

from search.analysis import analyze

@dataclass
class Abstract:
    ID: int
    abstract: str

    @property
    def fulltext(self):
        return self.abstract

    def analyze(self):
        self.term_frequencies = Counter(analyze(self.fulltext))

    def term_frequency(self, term):
        return self.term_frequencies.get(term, 0)
