class BitSet:
    def __init__(self, dimension, mask=0):
        self.dimension = dimension
        self.mask = mask

    @classmethod
    def from_subset(cls, dimension, subset):
        mask = 0
        for x in subset:
            if not 0 <= x <= dimension - 1:
                raise ValueError(f"Element {x} is out of range [0, {dimension - 1}]")
            mask |= 1 << x
        return cls(dimension, mask)

    def add(self, x):
        """Add element x (0 <= x <= dimension - 1)."""
        self.mask |= 1 << x

    def remove(self, x):
        """Remove element x."""
        self.mask &= ~(1 << x)

    def contains(self, x):
        """Check whether x is in the set."""
        return bool(self.mask & (1 << x))

    def union(self, other):
        return BitSet(self.dimension, self.mask | other.mask)

    def intersection(self, other):
        return BitSet(self.dimension, self.mask & other.mask)

    def difference(self, other):
        return BitSet(self.dimension, self.mask & ~other.mask)

    def elements(self):
        return [i for i in range(self.dimension) if self.mask & (1 << i)]

    def as_set(self):
        return set(self.elements())

    def __eq__(self, other):
        if not isinstance(other, BitSet):
            return NotImplemented
        return (
                self.dimension == other.dimension
                and self.mask == other.mask
        )

    def is_empty(self):
        return self.mask == 0

    def __repr__(self):
        return f"BitSet({self.elements()})"