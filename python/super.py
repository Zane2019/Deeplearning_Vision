

class FooParent(object):
    # parent =0
    def __init__(self):{
        print(" I am Parent!")
        # FooParent.parent+=1
    }
    def who(self):
        print("parent: ")

class Child(FooParent):
    def __init__(self):
        print(" I am Child")

class Child2(FooParent):
    def __init__(self):
        super().__init__()
        self.parent = 100
        print(" I am Child2")

if __name__ == "__main__":
    ch=Child()
    ch2=Child2()

    ch2.who()
