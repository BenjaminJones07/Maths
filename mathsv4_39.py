from abc import ABC, abstractmethod
from random import randrange
from typing import *
import numpy
import math

CalcType = "Maths"
ECalcReturn = CalcType
ICalcReturn = Tuple[ECalcReturn, bool]
Var = "Variable"

DELTA: float = 0.001

def Round(x: Any, n: int = 0) -> Any:
    if isinstance(x, complex):
        return complex(round(x.real, n), round(x.imag, n))
    return round(x, n)

def Key(x: Any) -> float:
    if isinstance(x, complex):
        return x.real
    return x

def Int(x: Any) -> Any:
    if isinstance(x, complex):
        return complex(int(x.real), int(x.imag))
    return int(x)

def floatToInt(x: Any) -> Any:
    return Int(x) if abs(x - Int(x)) == 0 else x

class Calculus(ABC):
    @abstractmethod
    def iderivative(self, respect: Var) -> ICalcReturn:
        raise NotImplementedError(f"iderivative method not implemented on type {type(self).__name__}")
    
    def derivative(self, respect: Optional[Var] = None) -> ECalcReturn:
        if not respect: respect: Var = X
        
        h: ICalcReturn = self.iderivative(respect)
        
        if h[1]: return h[0]
        return Zero()
    
class Expression(ABC):
    @abstractmethod
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        raise NotImplementedError(f"evaluate method not implemented on type {type(self).__name__}")
    
    @abstractmethod
    def degree(self, v: Optional[Var] = None) -> float:
        # Get the degree of the structure (default to x degree)
        pass
    
    def fSimplify(self) -> CalcType:
        # Fource the item to retry simplifying itself
        return self
    
    def numerator(self, respect: Var = "X") -> CalcType:
        return self
    
class Format(ABC):
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError(f"__str__ method not implemented on type {type(self).__name__}")
    
    @abstractmethod    
    def __repr__(self) -> str:
        return NotImplementedError(f"__repr__ method not implemented on type {type(self).__name__}")
    
    def __format__(self, __format_spec: str) -> str:
        return format(str(self), __format_spec)
    
class Maths(Calculus, Expression, Format):  
    def iapproxRoots(self, partial: "PartialEval", a: float) -> Optional[float]:
        # Iterate for given guess (Newton-Raphson method)
        values: List[float] = [a]
        dist: complex = complex(0, 0)
        
        while len(set(values)) == len(values):
            try: values.append(values[-1] - (partial(self, values[-1]) / partial(self.derivative(partial.store), values[-1])))
            except ZeroDivisionError:
                # Hacky fix, don't like
                return values[-1] if abs(partial(self, values[-1])) <= 0.001 else None
            
            tmp: Union[complex, float] = abs(values[-1] - values[-2])
            if tmp.real > dist.real and tmp.imag > dist.imag and len(values) > 2:
                return None # No root, jumping around (distance travelled should not increase)
            
            dist = tmp

        return values[-1]
    
    def approxRoots(
        s,
        partial: Optional["PartialEval"] = None,
        autoGuessing: bool = True,
        rounding: int = 4,
        sanitizeRange: float = 0.001,
        errorPercentage: float = 1.8,
        manualRange: Tuple[int, int] = (-10, 10)
    ) -> List[float]:
        if not partial: partial: PartialEval = PartialEval()
        
        self: CalcType = s.numerator()
        degree: float = self.degree(partial.store)
        
        if degree == 0: raise ValueError("The polynomial did not contain the variable")
        if degree == 1: return [floatToInt(round(-(partial(self, 0) / self.derivative(partial.store).val), rounding))]
        # Could add quadratic formula?
        
        else:
            roots: List[float] = list()
            
            if not autoGuessing:
                start, end = sorted(manualRange)
                tries: int = 0
                
                while tries < errorPercentage * (1.7 ** degree) and len(roots) != degree: # Allow 25% of attempts to be indistinctive
                    tries += 1
                    
                    root: Optional[float] = self.iapproxRoots(partial, randrange(start, end))
                    
                    if not root: continue
                    
                    for v in roots:
                        if abs(v - root) < sanitizeRange:
                            flag: bool = True
                            break
                        
                    if flag: continue
                    
                    roots.append(root)
                
            else:
                delta: complex = complex(DELTA, DELTA) # Arbitrary small ammount (𝛿)
                turnPoints: List[float] = self.derivative(partial.store).approxRoots(partial, True, rounding)
                print(turnPoints)

                # Start and end points don't allow for linear approximation
                neX: float = partial(self.derivative(partial.store).derivative(partial.store), turnPoints[0]) # Gradient for x to negative infinity (Change to use second derivative at x)
                # if (neX < 0) != (partial(self, turnPoints[0]) <= 0): # If m is negative, the turning point must be positive for there to be a root and vice versa
                root: Optional[float] = self.iapproxRoots(partial, turnPoints[0] - delta)
                if root is not None: roots.append(root)
                    
                for i in range(len(turnPoints) - 1):
                    a, b = turnPoints[i], turnPoints[i + 1]
                    if ((a.real <= 0) == (b.real < 0)) or ((a.imag <= 0) == (b.imag < 0)):
                        print("skip", a, b)
                        continue # No root between the points
                    guess: float = (b + a) / 2
                    root: Optional[float] = self.iapproxRoots(partial, guess)
                    if root is not None: roots.append(root)
                    
                poX: float = partial(self.derivative(partial.store).derivative(partial.store), turnPoints[-1]) # Gradient for x to positive infinity
                # if (poX < 0) != (partial(self, turnPoints[-1]) < 0): # If m is negative, the turning point must be positive for there to be a root and vice versa
                root: Optional[float] = self.iapproxRoots(partial, complex(turnPoints[-1].real + delta, turnPoints[-1].imag - delta))
                if root is not None: roots.append(root)
                
            roots: List[float] = sorted(list(map(lambda x: Round(x, rounding), roots)), key=Key)
            return list(map(floatToInt, roots))
    
    # partialEval = lambda e, v: e.evaluate(x=5, y=v, z=2)
    def approxInt(self, a: float, b: float, partial: Optional["PartialEval"] = None, strips: int = 1000) -> float:
        if not partial: partial: PartialEval = PartialEval()
        
        # Approximate definite integral (Trapezium Rule or Simpson's Rule)
        if not strips % 2 == 0: raise ValueError("Number of strips was not even")

        h: float = (b - a) / strips
        values: List[float] = list()
        while a != b:
            values.append(partial(self, a))
            a = round(a + h, 6)
        values.append(partial(self, b))
            
        startend: int = values.pop(0) + values.pop(-1)
        
        odd, even = float(0), float(0)
        for (i, v) in enumerate(values):
            if i % 2 == 1: even += v
            else: odd += v
        
        return h * (startend + 4 * odd + 2 * even) / 3

# Values

class Value(Maths):
    def __init__(self, val: float, disp: Optional[str] = None) -> None:
        self.val: float = val
        self.disp: Optional[str] = disp
        
    def iderivative(self, respect: Var) -> ICalcReturn:
        return (Zero(), False)
    
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return self.val
    
    def degree(self, v: Optional[Var] = None) -> float:
        return 0
    
    def __str__(self) -> str:
        return f"{self.val if not self.disp else self.disp}"
    
    def __repr__(self) -> str:
        return f"Value({self.val}{f', {self.disp}' if self.disp else ''})"
    
    def __mul__(self, other) -> float:
        return self.val * other
    
    __rmul__ = __mul__

class Zero(Value):
    def __init__(self) -> None:
        super().__init__(0)
        
class One(Value):
    def __init__(self) -> None:
        super().__init__(1)

class Pi(Value):
    def __init__(self) -> None:
        super().__init__(round(math.pi, 3), "π")
        
class Euler(Value):
    def __init__(self) -> None:
        super().__init__(round(math.e, 3), "e")

# Variables

class Variable(Maths):
    def iderivative(self, respect: Var) -> ICalcReturn:
        if not respect is type(self): return (self, False)
        return (One(), True)
    
    def degree(self, v: Optional[Var] = None) -> float:
        if not v: v: Var = X
        return 1 if isinstance(self, v) else 0
    
    def __str__(self) -> str:
        return type(self).__name__.lower()
    
    def __repr__(self) -> str:
        return f"{str(self).upper()}()"
    
class X(Variable):
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return x
    
class Y(Variable):
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return y
    
class Z(Variable):
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return z

# PartialEval

class PartialEval:
    def __init__(self, x: float = None, y: float = None, z: float = None) -> None:
        params = [t is None for t in [x, y, z]]
        if params in [[True] * 3, [True, False, False]]:
            self.store: Var = X
            self.callable: Callable = lambda e, v: e.evaluate(v, y, z)
        elif params == [False, True, False]:
            self.store: Var = Y
            self.callable: Callable = lambda e, v: e.evaluate(x, v, z)
        elif params == [False, False, True]:
            self.store: Var = Z
            self.callable: Callable = lambda e, v: e.evaluate(x, y, v)
        else:
            raise ValueError("One value must be omitted")
            
    def __call__(self, e: "Maths", v: float) -> float:
        return self.callable(e, v)

# Structures

class Term(Maths):
    def __new__(cls: "Term", h: Maths, expo: float) -> CalcType:
        if isinstance(h, Value):
            return Value(h.val ** expo)
        
        if isinstance(h, Mult):
            for t in h.holds: expo *= Term(t, 1).expo
            return Mult(h.coef ** expo, *h.holds)
        
        return super().__new__(cls)
    
    def __init__(self, h: Maths, expo: float) -> None:
        if isinstance(h, Term):
            expo *= h.expo
            h = h.holds
        
        self.holds: Maths = h
        self.expo: float = expo
        
    def iderivative(self, respect: Var) -> ICalcReturn:
        h: ICalcReturn = self.holds.iderivative(respect)
        
        if h[1]:
            return (
                Mult(
                    self.expo,
                    Term(h[0], 1),
                    Term(self.holds, self.expo - 1)
                ),
                True
            )
        
        return (self, False)
    
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return self.holds.evaluate(x, y, z) ** self.expo
    
    def degree(self, v: Optional[Var] = None) -> float:
        if not v: v: Var = X
        return self.expo * self.holds.degree(v)
    
    def numerator(self) -> CalcType:
        return self if self.expo >= 0 else Value(1)
    
    def fSimplify(self) -> CalcType:
        if self.expo == 0:
            return One()
        if self.expo == 1:
            return self.holds
        return self
    
    def __str__(self) -> str:
        if self.expo == 0: return "1"
        
        out: str = str(self.holds)
        if self.expo == 1: return out
        
        if isinstance(self.holds, Poly): out = f"({out})"
        return out + f"^{self.expo}"
    
    def __repr__(self) -> str:
        return f"Term({self.holds!r}, {self.expo!r})"
    
def normalize(l: List[Maths]) -> Tuple[float, List[Term]]:
    coef: float = 1
    terms: List[Term] = list()
    
    for e in l:
        if isinstance(e, Value):
            coef *= e.val
        
        elif isinstance(e, Mult):
            coef *= e.coef
            terms.extend(list(map(lambda x: Term(x, 1), e.holds)))
            
        else: # Variable, Term, Poly, Func
            terms.append(Term(e, 1))
            
    return (coef, terms)
    
class Mult(Maths):
    def __new__(cls: "Mult", coef: float, *args: Maths) -> CalcType:
        args = list(map(lambda x: x.fSimplify(), args))
        if len(args) == 0: return Value(coef)
        if coef == 1 and len(args) == 1: return args[0]
        
        nonValue: bool = False
        for arg in args:
            if not isinstance(arg, Value):
                nonValue = True
                break
            
        if not nonValue:
            return Value(coef * math.prod(list(map(lambda x: x.val, args))))
        
        return super().__new__(cls)
    
    def __init__(self, coef: float, *args: Maths) -> None:
        # Normalize
        innerCoef, terms = normalize(args)
        
        # Group
        dTerms: Dict[str, float] = dict()
        for t in terms:
            dTerms[repr(t.holds)] = dTerms.get(repr(t.holds), 1) * t.expo
        terms: List[Term] = [Term(eval(r), e) for r, e in dTerms.items()]
        
        self.coef: float = coef * innerCoef
        self.holds: List[Term] = list()
        
        # Simplify
        for t in terms:
            t = t.fSimplify()
            if isinstance(t, Value):
                self.coef *= t.val
            else:
                self.holds.append(t)
                
    def iderivative(self, respect: Var) -> ICalcReturn:
        dList: List[ICalcReturn] = list(map(lambda x: x.iderivative(respect), self.holds))
        
        consts: List[ICalcReturn] = list()
        derive: List[Maths] = list()
        derived: List[Maths] = list()
        for i, v in enumerate(dList):
            if v[1]:
                derived.append(v[0])
                derive.append(self.holds[i])
            else:
                consts.append(v[0])
        
        p: Poly = Poly(
            *[
                Mult(1, v, *[t for z, t in enumerate(derive) if z != i]) for i, v in enumerate(derived)
            ]
        )
        
        return (
            Mult(
                self.coef,
                *consts,
                p
            ),
            len(derived) > 0
        )
    
    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return self.coef * math.prod([t.evaluate(x, y, z) for t in self.holds])
    
    def degree(self, v: Optional[Var] = None) -> float:
        if not v: v: Var = X
        return max([t.degree(v) for t in self.holds])
    
    def numerator(self) -> CalcType:
        return Mult(self.coef, *[t for t in self.holds if t.degree(Variable) > 0])
    
    def __str__(self) -> str:
        terms: List[str] = list(filter(lambda x: x != "1", map(str, sorted(self.holds, key=lambda t: t.degree(), reverse=True))))
        if "0" in terms or self.coef == 0: return "0"
        terms = [f"({t})" if "^" in t else t for t in terms]
        if self.coef != 1: terms = [str(self.coef), *terms]
        
        return ''.join(terms)
    
    def __repr__(self) -> str:
        return f"Mult({self.coef}, {', '.join(list(map(repr, self.holds)))})"
    
class Poly(Maths):
    def __new__(cls: "Poly", *args: Maths) -> CalcType:
        if len(args) == 0: return Zero()
        if len(args) == 1: return args[0]
        return super().__new__(cls)
    
    def __init__(self, *args: Maths) -> None:
        # Normalization
        items: List[Maths] = list()
        for item in args:
            if isinstance(item, Poly):
                items.extend(item.holds)
            else:
                items.append(item)
                
        # Grouping (Complex)
        values: float = 0
        tmpDict: Dict[str, float] = dict()
        for item in items:
            if isinstance(item, Value):
                values += item.val
            
            else:
                tmpDict[repr(item)] = tmpDict.get(repr(item), 0) + 1
                
        self.holds: List[Maths] = list()
        if values != 0: self.holds.append(Value(values))
        for e, c in tmpDict.items():
            self.holds.append(Mult(c, eval(e)))
            
    def iderivative(self, respect: Var) -> ICalcReturn:
        h: List[ICalcReturn] = [t.iderivative(respect) for t in self.holds]
        h: List[CalcType] = list(map(lambda x: x[0], filter(lambda x: x[1], h)))
        
        return (
            Poly(
                *h
            ),
            len(h) > 0
        )

    def evaluate(self, x: float = 0, y: float = 0, z: float = 0) -> float:
        return sum([t.evaluate(x, y, z) for t in self.holds])
    
    def degree(self, v: Optional[Var] = None) -> float:
        if not v: v: Var = X
        return max([t.degree(v) for t in self.holds])
    
    def __str__(self) -> str:
        return ' + '.join(list(map(str, sorted(self.holds, key=lambda t: t.degree(), reverse=True))))
    
    def __repr__(self) -> str:
        return f"Poly({', '.join(list(map(repr, self.holds)))})"
    
if __name__ == "__main__":
    t: Term = Term(Euler(), 2)
    print(f"d({t})/dx = {t.derivative()}")

    print()

    t1: Term = Term(Y(), 4)
    print(f"The highest degree of x in {t1} is {t1.degree()}")
    print(f"The highest degree of y in {t1} is {t1.degree(Y)}")
    print(f"d({t1})/dy = {t1.derivative(Y)}")
    print(f"The highest degree of y in d({t1})/dy is {t1.derivative(Y).degree(Y)}")

    print()

    terms: List[Term] = [
        Term(t1, 2),
        Term(Y(), 4),
        Term(X(), 8),
        Term(Z(), 1),
        One(),
        Value(2)
    ]
    
    m: Mult = Mult(1, *terms)
    
    print(f"d({m})/dx = {m.derivative()}")

    print()

    p: Poly = Poly(*terms)
    
    print(f"d({p})/dx = {p.derivative()}")
    print(f"d({p})/dy = {p.derivative(Y)}")
    print(f"d({p})/dz = {p.derivative(Z)}")
    
    print()
    
    test = Mult(
        1,
        Y(),
        Term(
            Poly(
                Value(1),
                Term(Y(), 2)
            ),
            -1
        )
    )
    
    print(f"Definite integral of {test} from 0 to 4 ≈ {test.approxInt(0, 4, PartialEval(x=0, z=0))}")
    print(f"{test} = 0 => x ∈ {test.approxRoots(PartialEval(x=0, z=0))}")
    
    print()
    
    test2: Poly = Poly(
        Term(
            X(),
            1
        ),
        Value(3)
    )
    
    print(f"{test2} = 0 => x ∈ {test2.approxRoots()}")
    
    print()
    
    polynomial: Poly = Poly(
        Term(
            X(),
            3
        ),
        Mult(
            4,
            Term(
                X(),
                2
            )
        ),
        Value(
            -5
        )
    )
    
    print(f"{polynomial} = 0 => x ∈ {polynomial.approxRoots()}")
    
    print()
    
    quartic: Poly = Poly(
        Term(
            X(),
            4
        ),
        Mult(
            -3,
            Term(
                X(),
                3
            )
        ),
        Mult(
            3,
            X()
        ),
        Value(
            -0.5
        )
    )
    
    print(f"{quartic} = 0 => x ∈ {quartic.approxRoots()}")
    
    print()
    
    cubed: Term = Term(
        Poly(
            X(),
            Value(-5)
        ),
        3
    )
    
    print(f"{cubed} = 0 => x ∈ {cubed.approxRoots()}")