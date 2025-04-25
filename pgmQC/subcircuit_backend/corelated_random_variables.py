

class RandomVariable:
    def __init__(self, mean, variance, identifier):
        self.mean = mean
        self.variance = variance
        self.identifier = identifier
        self.eigenvalues = None
    
    def assign_eigenvalues(self, eigenvalues):
        self.eigenvalues = eigenvalues
    
    def identifier_to_index(self):
        ret = 0
        for symbol in self.identifier:
            if symbol == '0' or symbol == 'I':
                ret = ret * 4 + 0
            elif symbol == '1' or symbol == 'X':
                ret = ret * 4 + 1
            elif symbol == '+' or symbol == 'Y':
                ret = ret * 4 + 2
            elif symbol == 'i' or symbol == 'Z':
                ret = ret * 4 + 3
        return ret
    
    # Overload the + operator for adding two random variables
    def __add__(self, other):
        if isinstance(other, RandomVariable):
            # New mean is the sum of the means
            new_mean = self.mean + other.mean
            # New variance is the sum of the variances (assuming independence)
            new_variance = self.variance + other.variance
            return RandomVariable(new_mean, new_variance, f"({self.identifier} + {other.identifier})")
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'RandomVariable' and '{type(other).__name__}'")
        # Overload the * operator for multiplying by a scalar
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Scalar multiplication
            new_mean = self.mean * other
            new_variance = self.variance * (other ** 2)
            return RandomVariable(new_mean, new_variance, f"({other} * {self.identifier})")
        
        elif isinstance(other, RandomVariable):
            # Multiplying two random variables assuming independence
            new_mean = self.mean * other.mean
            new_variance = (
                (self.mean ** 2) * other.variance +
                (other.mean ** 2) * self.variance +
                self.variance * other.variance
            )
            return RandomVariable(new_mean, new_variance, f"({self.identifier} * {other.identifier})")
        
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'RandomVariable' and '{type(other).__name__}'")

    # Ensure multiplication works both ways (commutative with scalar)
    __rmul__ = __mul__

class CorelatedRandomVariables:
    def __init__(self, random_variables):
        self.random_variables = random_variables
        self.groups = []
        self.covariance = {}
