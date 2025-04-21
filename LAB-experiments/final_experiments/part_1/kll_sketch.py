import random
import math


class KLL:
    def __init__(self, k):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k  # Base buffer size
        self.compactors = []
        self.H = 0  # Number of levels
        self.size = 0  # Total number of stored elements
        self.maxSize = 0  # Computed total max capacity
        self._initialize_compactors()

    def _initialize_compactors(self):
        """Initialize the first compactor and compute initial capacity."""
        self.compactors.append(Compactor())
        self.H = 1  # Ensure H starts at 1
        self._update_maxSize()

    def _update_maxSize(self):
        """Recomputes the total storage capacity based on the paper's exponential decay formula."""
        self.maxSize = sum(self._capacity(h) for h in range(self.H))

    def _capacity(self, height):
        """Computes the capacity for each level based on exponential decay."""
        depth = self.H - height - 1
        return max(2, int(math.ceil(self.k * (2 / 3) ** depth)))

    def insert(self, item):
        """Inserts a new item and manages compaction if required."""
        self.size += 1  # Always increment size
        self.compactors[0].append(item)

        # If we're still below k items in the bottom level, do not compress yet.
        if len(self.compactors[0]) <= self.k:
            return

        # Otherwise, check if we exceed capacity and need to compress
        if len(self.compactors[0]) >= self._capacity(0):
            self._compress()

    def _compress(self):
        """Compresses the sketch following the compaction strategy in the paper."""
        for h in range(self.H):
            if len(self.compactors[h]) >= self._capacity(h):
                if h + 1 >= self.H:
                    self._grow()
                promoted_items = self.compactors[h].compact()
                self.compactors[h + 1].extend(promoted_items)
                self.size = sum(len(c) for c in self.compactors)

    def _grow(self):
        """Adds a new compactor level when needed."""
        self.compactors.append(Compactor())
        self.H = len(self.compactors)
        self._update_maxSize()

    def merge(self, other):
        # Ensure both sketches have matching heights
        while self.H < other.H:
            self._grow()
        while self.H > other.H:
            other._grow()

        # Merge compactor data
        for h in range(other.H):
            self.compactors[h].extend(other.compactors[h])

        self.size += other.size
        self._update_maxSize()

        # Ensure compactions occur if needed
        while self.size >= self.maxSize:
            self._compress()

    def rank(self, value):
        """Returns the approximate rank of a value."""
        r = 0
        for h, c in enumerate(self.compactors):
            for item in c:
                if item <= value:
                    r += 2**h
        return r

    def cdf(self):
        """Computes the approximate CDF from the sketch."""
        items_and_weights = []
        for h, c in enumerate(self.compactors):
            items_and_weights.extend((item, 2**h) for item in c)
        items_and_weights.sort()
        total_weight = sum(weight for _, weight in items_and_weights)
        cumulative_weight = 0
        cdf_values = []
        for item, weight in items_and_weights:
            cumulative_weight += weight
            cdf_values.append((item, cumulative_weight / total_weight))
        return cdf_values


class Compactor(list):
    def __init__(self):
        """Initializes a compactor with an empty list and compaction counter."""
        super().__init__()
        self.numCompactions = 0

    def compact(self):
        """Performs a compaction operation following the randomized strategy from the paper."""
        self.sort()
        offset = random.randint(0, 1)  # Randomly select even or odd indices
        compacted_items = [self[i] for i in range(offset, len(self), 2)]
        self.clear()  # Strictly clear list after compaction
        self.numCompactions += 1
        return compacted_items
