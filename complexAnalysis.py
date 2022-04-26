from mathsv4_39 import *

imagQuart: Poly = Poly(
    Mult(
        3,
        Term(
            X(),
            4
        )
    ),
    Mult(
        2,
        Term(
            X(),
            3
        )
    ),
    Mult(
        -5,
        Term(
            X(),
            2
        )
    ),
    Mult(
        -3,
        X()
    ),
    Value(3)
)

print(imagQuart)

def analyse(z: complex) -> None:
    print(f"z = {z}")
    print(f"f(z) = {Round(imagQuart.evaluate(z), 3)}")
    print(f"f'(z) = {Round(imagQuart.derivative().evaluate(z), 3)}")
    print(f"f''(z) = {Round(imagQuart.derivative().derivative().evaluate(z), 3)}")

    try:
        move: complex = -Round(Round(imagQuart.evaluate(z), 3) / Round(imagQuart.derivative().evaluate(z), 3), 3)
    except ZeroDivisionError:
        move: str = "undefined"
    
    print(f"-f(z)/f'(z) = {move}")
    
def testWaters(z: complex, delta: float) -> List[List[bool]]: # [-ve +i 0 -i, +ve +i 0 -i]
    h: complex = imagQuart.evaluate(z)
    def test(z: complex) -> bool:
        print(Round(z, 3), Round(h, 3))
        reals: bool = (((z.real < 0) != (h.real < 0)) or ((z.real == 0) or (h.real == 0)))
        imags: bool = (((z.imag < 0) != (h.imag < 0)) or ((z.imag == 0) or (h.imag == 0)))
        return reals and imags
    
    # -ve test
    neg: List[bool] = [test(imagQuart.derivative().evaluate(complex(z.real - delta, z.imag - (i * delta)))) for i in range(-1, 2)]
    
    # +ve test
    pos: List[bool] = [test(imagQuart.derivative().evaluate(complex(z.real + delta, z.imag - (i * delta)))) for i in range(-1, 2)]
    
    return [neg, pos]
    
while True:
    print("Options:")
    print("1) Analyse values of the zeroth, first, second, and third derivatives of f(x) for a single complex value")
    print("2) Analyse values of the zeroth, first, second, and third derivatives of f(x) with delta value")
    print("3) Analyse output of Newton-Raphson method for complex input")
    print("4) Test the waters for complex input")
    
    try:
        choice: int = int(input("> "))
        re: float = float(input("Re(z) = "))
        im: float = float(input("Im(z) = "))
    except ValueError:
        continue
    
    z: complex = complex(re, im)
    print()
    
    if choice == 1:
        analyse(z)
        input()
    
    elif choice == 2:
        try:
            d: float = float(input("𝛿 ="))
        except ValueError:
            continue
        
        analyse(z)
        input()

        analyse(z + complex(d, d)) # + +
        input()
        
        analyse(z + complex(d, 0)) # + 0
        input()

        analyse(z + complex(d, -d)) # + -
        input()

        analyse(z - complex(d, -d)) # - +
        input()
        
        analyse(z - complex(d, 0)) # - 0
        input()

        analyse(z - complex(d, d)) # - -
        input()
        
    elif choice == 3:
        print(Round(imagQuart.iapproxRoots(PartialEval(y=0, z=0), z), 3))
        input()
        
    elif choice == 4:
        try:
            d: float = float(input("𝛿 ="))
        except ValueError:
            continue
        
        print(testWaters(z, d))
