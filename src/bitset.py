class BitSet:
    def __init__(self, dimension, mask=0):
        self.dimension = dimension
        self.mask = mask

    @classmethod
    def from_subset(cls, dimension, subset):
        mask = 0
        for x in subset:
            if not 1 <= x <= dimension:
                raise ValueError(f"Feature {x} is bigger than the number of features {dimension}")
            mask |= 1 << (x - 1)
        return cls(dimension, mask)

    def add(self, x):
        """Add element x (1 <= x <= d)."""
        self.mask |= 1 << (x - 1)

    def remove(self, x):
        """Remove element x."""
        self.mask &= ~(1 << (x - 1))

    def contains(self, x):
        """Check whether x is in the set."""
        return bool(self.mask & (1 << (x - 1)))

    def union(self, other):
        return BitSet(self.dimension, self.mask | other.mask)

    def intersection(self, other):
        return BitSet(self.dimension, self.mask & other.mask)

    def difference(self, other):
        return BitSet(self.dimension, self.mask & ~other.mask)

    def elements(self):
        return [i + 1 for i in range(self.dimension)
                if self.mask & (1 << i)]

    def as_set(self):
        return set(self.elements())


    def __repr__(self):
        return f"BitSet({self.elements()})"