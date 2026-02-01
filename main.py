from audioop import error
def main():
    print("Hello from research-experiment!")
    acutual = new_feature()
    expected = 1
    
    test(expected,acutual)


def new_feature():
    return 1

def test(expected, acutual):
    if expected != acutual:
        error

if __name__ == "__main__":
    main()
